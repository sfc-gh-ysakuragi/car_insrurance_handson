# ------------------------------------------------------------
# Streamlit in Snowflake: Cortex Search RAG チャットUI版
#  - チャットバブル（st.chat_message / st.chat_input）
#  - 履歴ウィンドウ（直近k往復を文脈へ）
#  - 参照PDF/URL/チャンクの可視化（各回答内にエクスパンダ）
#  - 履歴クリアボタン/モデル選択/サービス選択
#  - DB/スキーマ選択（実在名の選択リスト）
#  - 任意カラム=値フィルタ（DISTINCT 値の選択、未取得時は手入力）
# ------------------------------------------------------------
# 必要パッケージ（Streamlit Packagesに追加）:
# - snowflake>=0.8.0
# - snowflake-ml-python
# ------------------------------------------------------------
# Made by Sakuragi (Snowflake)
# - 2025/10/06 Final Update
# ------------------------------------------------------------

from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.core import Root
from snowflake.cortex import Complete

# 設定
DB_HINT_DEFAULT = "CORTEX_SEARCH_SAMPLE"
MODELS = [
    "claude-4-sonnet",
    "claude-3-7-sonnet",
    "claude-3-5-sonnet",
    "llama4-maverick",
    "llama4-scout",
]
MANUAL_SENTINEL = "<手入力>"

# Snowflake接続
session = get_active_session()
root = Root(session)


# ---------- ユーティリティ ----------

def df_rows_to_dicts(rows, cols) -> List[Dict[str, Any]]:
    def norm(k: str) -> str:
        s = str(k or "").strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        return s.lower()
    return [{norm(cols[i]): r[i] for i in range(len(cols))} for r in rows]


def build_fq_name(db: str, schema: str, name: str) -> str:
    n = (name or "").strip()
    if "." in n:
        return n
    n = n.strip('"')
    return f'{db}.{schema}."{n}"'


def quote_ident(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return s
    if s.startswith('"') and s.endswith('"'):
        return s
    return f'"{s}"'


def list_databases() -> List[str]:
    df = session.sql("SHOW DATABASES;")
    rows = df.collect()
    cols = [f.name for f in df.schema.fields]
    dicts = df_rows_to_dicts(rows, cols)
    names: List[str] = []
    for rd in dicts:
        n = (rd.get("name") or rd.get("database_name") or "").strip()
        if n and n not in names:
            names.append(n)
    return names


def list_schemas(db: Optional[str]) -> List[str]:
    if not db:
        return []
    df = session.sql(f"SHOW SCHEMAS IN DATABASE {quote_ident(db)};")
    rows = df.collect()
    cols = [f.name for f in df.schema.fields]
    dicts = df_rows_to_dicts(rows, cols)
    names: List[str] = []
    for rd in dicts:
        n = (rd.get("name") or rd.get("schema_name") or "").strip()
        if n and n not in names:
            names.append(n)
    return names


def load_cortex_services(db_hint: Optional[str] = None, schema_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    queries = []
    if db_hint and schema_hint:
        queries.append(f"SHOW CORTEX SEARCH SERVICES IN SCHEMA {quote_ident(db_hint)}.{quote_ident(schema_hint)};")
    if db_hint:
        queries.append(f"SHOW CORTEX SEARCH SERVICES IN DATABASE {quote_ident(db_hint)};")
    queries.append("SHOW CORTEX SEARCH SERVICES;")
    queries.append("SHOW CORTEX SEARCH SERVICES IN ACCOUNT;")

    services: List[Dict[str, Any]] = []
    seen = set()

    for q in queries:
        df = session.sql(q)
        rows = df.collect()
        cols = [f.name for f in df.schema.fields]
        dicts = df_rows_to_dicts(rows, cols)

        for rd in dicts:
            db = (rd.get("database_name") or rd.get("database") or "").strip()
            schema = (rd.get("schema_name") or rd.get("schema") or "").strip()
            name = (rd.get("name") or "").strip()
            if not name:
                continue

            fq_name = build_fq_name(db, schema, name)
            key = (db, schema, fq_name)
            if key in seen:
                continue
            seen.add(key)

            # 大文字/小文字は保持（フィルタ列名に使用）
            search_col = (rd.get("search_column") or "chunk")
            cols_raw = (rd.get("columns") or "")
            cols_avail = [c.strip() for c in cols_raw.split(",") if c.strip()]

            services.append({
                "fq_name": fq_name,
                "db": db,
                "schema": schema,
                "short_name": name.strip('"'),
                "search_column": search_col,
                "columns_available": cols_avail,
            })

        if services:
            break

    return services


def intersect_preserving_order_caseaware(candidates: List[str], allowed: List[str]) -> List[str]:
    allowed_map = {a.lower(): a for a in allowed}
    out: List[str] = []
    seen: set[str] = set()
    for c in candidates:
        lc = c.lower()
        if lc in allowed_map:
            actual = allowed_map[lc]
            if actual not in seen:
                out.append(actual)
                seen.add(actual)
    return out


def describe_cortex_service_properties(db: str, schema: str, short_name: str) -> Dict[str, str]:
    try:
        fq = f"{quote_ident(db)}.{quote_ident(schema)}.{quote_ident(short_name)}"
        df = session.sql(f"DESCRIBE CORTEX SEARCH SERVICE {fq};")
        rows = df.collect()
        cols = [f.name for f in df.schema.fields]
        dicts = df_rows_to_dicts(rows, cols)
        props: Dict[str, str] = {}
        for rd in dicts:
            key = (rd.get("property") or rd.get("key") or rd.get("name") or "").strip().upper()
            val = (rd.get("value") or rd.get("property_value") or rd.get("text") or "").strip()
            if key:
                props[key] = val
        return props
    except Exception:
        return {}


def get_distinct_values_for_column(meta: Dict[str, Any], column: str, max_values: int = 200) -> List[str]:
    # 1) DESCRIBE 情報からターゲットを推定して DISTINCT
    if not meta or not column:
        return []
    db = meta.get("db", "")
    schema = meta.get("schema", "")
    short_name = meta.get("short_name", "") or meta.get("fq_name", "").split(".")[-1]
    props = describe_cortex_service_properties(db, schema, short_name)

    source_table = props.get("TARGET_TABLE") or props.get("SOURCE_TABLE") or ""
    query_text = props.get("QUERY") or props.get("STATEMENT_TEXT") or ""

    col = quote_ident(column)
    sql = ""
    if source_table:
        sql = f"SELECT DISTINCT TO_VARCHAR({col}) AS V FROM {source_table} WHERE {col} IS NOT NULL ORDER BY 1 LIMIT {max_values}"
    elif query_text:
        qt = query_text.strip().rstrip(";")
        sql = f"WITH BASE AS ({qt}) SELECT DISTINCT TO_VARCHAR({col}) AS V FROM BASE WHERE {col} IS NOT NULL ORDER BY 1 LIMIT {max_values}"

    if sql:
        try:
            df = session.sql(sql)
            rows = df.collect()
            values = []
            for r in rows:
                v = r[0]
                if v is None:
                    continue
                s = str(v).strip()
                if s not in values:
                    values.append(s)
            if values:
                return values
        except Exception:
            pass

    # 2) フォールバック: サービス検索のサンプルから推定（上位N件）
    return get_distinct_values_via_search(meta, column, sample_size=max_values)


def get_distinct_values_via_search(meta: Dict[str, Any], column: str, sample_size: int = 200) -> List[str]:
    try:
        db = meta.get("db", "")
        schema = meta.get("schema", "")
        short_name = meta.get("short_name", "") or meta.get("fq_name", "").split(".")[-1]
        svc = root.databases[db].schemas[schema].cortex_search_services[short_name]

        # いくつかのクエリで試行（空/ワイルドカード/汎用語）
        queries_to_try = ["", "*", "a", "the", "の"]
        seen: set[str] = set()
        values: List[str] = []

        for q in queries_to_try:
            try:
                doc = svc.search(q, columns=[column], limit=sample_size)
                for r in doc.results:
                    val = r.get(column) or r.get(column.lower()) or r.get(column.upper())
                    if val is None:
                        continue
                    s = str(val).strip()
                    if s and s not in seen:
                        seen.add(s)
                        values.append(s)
                        if len(values) >= sample_size:
                            break
                if values:
                    break
            except Exception:
                continue

        return values
    except Exception:
        return []


# ---------- サイドバー ----------

def init_sidebar():
    st.sidebar.header("設定")

    # DB/スキーマ 選択（実在名を取得して選択肢化）
    if "db_hint" not in st.session_state:
        st.session_state.db_hint = DB_HINT_DEFAULT or ""
    if "schema_hint" not in st.session_state:
        st.session_state.schema_hint = ""

    databases = list_databases()
    db_options = [""] + databases
    default_db = st.session_state.db_hint if st.session_state.db_hint in db_options else (
        DB_HINT_DEFAULT if DB_HINT_DEFAULT in databases else ""
    )
    chosen_db = st.sidebar.selectbox(
        "データベース",
        db_options,
        index=db_options.index(default_db) if default_db in db_options else 0,
        format_func=lambda v: "（指定しない）" if v == "" else v,
    )
    st.session_state.db_hint = chosen_db or ""

    schemas = list_schemas(chosen_db) if chosen_db else []
    schema_options = [""] + schemas
    default_schema = st.session_state.schema_hint if st.session_state.schema_hint in schema_options else ""
    chosen_schema = st.sidebar.selectbox(
        "スキーマ",
        schema_options,
        index=schema_options.index(default_schema) if default_schema in schema_options else 0,
        format_func=lambda v: "（指定しない）" if v == "" else v,
    )
    st.session_state.schema_hint = chosen_schema or ""

    # サービス選択
    services = load_cortex_services(
        db_hint=st.session_state.db_hint or None,
        schema_hint=st.session_state.schema_hint or None
    )
    st.sidebar.caption(f"検出サービス数: {len(services)}")

    if not services:
        st.sidebar.warning("サービスが見つかりません。完全修飾名の手入力も利用できます。")
        manual = st.sidebar.text_input('完全修飾名（例: CORTEX_SEARCH_SAMPLE.PUBLIC."JPI_SEARCH_SERVICE"）')
        if manual:
            parts = manual.split(".")
            st.session_state.selected_cortex_search_service = manual
            st.session_state.selected_cortex_meta = {
                "fq_name": manual,
                "db": parts[0] if len(parts) > 0 else "",
                "schema": parts[1] if len(parts) > 1 else "",
                "short_name": parts[-1].strip('"'),
                "search_column": "chunk",
                "columns_available": ["chunk", "relative_path", "file_url", "language"],
            }
        return

    options = [s["fq_name"] for s in services]
    chosen = st.sidebar.selectbox("Cortex Search Service（完全修飾名）", options, index=0)
    st.session_state.selected_cortex_search_service = chosen
    st.session_state.selected_cortex_meta = next((s for s in services if s["fq_name"] == chosen), None)

    st.sidebar.divider()
    # モデル選択
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = MODELS[0]
    st.session_state.selected_model = st.sidebar.selectbox(
        "回答モデル", MODELS, index=MODELS.index(st.session_state.selected_model)
    )

    st.sidebar.divider()

    # フィルタ（任意カラム=値, 値は DISTINCT 実データ or 手入力）
    meta = st.session_state.get("selected_cortex_meta") or {}
    columns_available: List[str] = meta.get("columns_available", []) or []

    if "filter_enabled" not in st.session_state:
        st.session_state.filter_enabled = False
    if "filter_column" not in st.session_state:
        st.session_state.filter_column = columns_available[0] if columns_available else ""
    else:
        if columns_available and st.session_state.filter_column not in columns_available:
            st.session_state.filter_column = columns_available[0]

    if "filter_value_options" not in st.session_state:
        st.session_state.filter_value_options = []
    if "filter_value_selected_key" not in st.session_state:
        st.session_state.filter_value_selected_key = ""
    if "filter_value_manual" not in st.session_state:
        st.session_state.filter_value_manual = ""

    st.session_state.filter_enabled = st.sidebar.toggle(
        "フィルタを使う（@eq）",
        value=st.session_state.filter_enabled,
        help="選択した列が値と完全一致のデータのみ検索します。"
    )
    if st.session_state.filter_enabled:
        st.session_state.filter_column = st.sidebar.selectbox(
            "フィルタ列",
            options=columns_available if columns_available else [""],
            index=(columns_available.index(st.session_state.filter_column) if st.session_state.filter_column in columns_available else 0),
            disabled=not columns_available,
            help="Cortex Search Serviceが保持するメタデータ列から選択"
        )

        # サービス/列の変更検知
        svc_key = st.session_state.selected_cortex_search_service or ""
        col_key = st.session_state.filter_column or ""
        changed = False
        if st.session_state.get("_last_service_key") != svc_key:
            st.session_state["_last_service_key"] = svc_key
            changed = True
        if st.session_state.get("_last_filter_column") != col_key:
            st.session_state["_last_filter_column"] = col_key
            changed = True

        # 値の再取得
        refresh_clicked = st.sidebar.button("値を再取得")
        if changed or refresh_clicked:
            st.session_state.filter_value_options = get_distinct_values_for_column(meta, st.session_state.filter_column)
            # 選択キー初期化
            st.session_state.filter_value_selected_key = ""
            st.session_state.filter_value_manual = ""

        opts = ["", MANUAL_SENTINEL] + st.session_state.filter_value_options
        st.session_state.filter_value_selected_key = st.sidebar.selectbox(
            "フィルタ値（DISTINCT 上位200件）",
            options=opts,
            index=opts.index(st.session_state.filter_value_selected_key) if st.session_state.filter_value_selected_key in opts else 0,
            format_func=lambda v: "（指定しない）" if v == "" else (v if v != MANUAL_SENTINEL else "＜手入力＞"),
        )

        if st.session_state.filter_value_selected_key == MANUAL_SENTINEL:
            st.session_state.filter_value_manual = st.sidebar.text_input(
                "手入力フィルタ値",
                value=st.session_state.filter_value_manual,
                placeholder="例: Japanese",
            )

        if not columns_available:
            st.sidebar.info("このサービスに列情報が見つかりません。サービス定義をご確認ください。")
        elif not st.session_state.filter_value_options and st.session_state.filter_value_selected_key != MANUAL_SENTINEL:
            st.sidebar.info("候補値が取得できませんでした。必要に応じて＜手入力＞を使ってください。")

    st.sidebar.divider()

    # 履歴
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # 各要素: {"timestamp","question","answer","model","contexts":[...]}

    # 参照チャンク数
    if "num_retrieved_chunks" not in st.session_state:
        st.session_state.num_retrieved_chunks = 5
    st.session_state.num_retrieved_chunks = st.sidebar.slider(
        "参照チャンク数", 1, 10, st.session_state.num_retrieved_chunks
    )

    st.sidebar.divider()

    # 履歴ウィンドウ（直近 k 往復）
    if "history_k" not in st.session_state:
        st.session_state.history_k = 3
    st.session_state.history_k = st.sidebar.slider(
        "過去履歴の参照数（往復）", 0, 10, st.session_state.history_k
    )

    # クリアボタン
    if st.sidebar.button("チャット履歴をクリア"):
        st.session_state.chat_history = []
        st.sidebar.success("履歴をクリアしました。")


# ---------- 検索/プロンプト ----------

def query_cortex_search_service(query: str, columns: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None):
    columns = columns or ["chunk", "file_url", "relative_path"]
    filter = filter or {}

    meta = st.session_state.selected_cortex_meta or {}
    db = meta.get("db", "")
    schema = meta.get("schema", "")
    short_name = meta.get("short_name", "") or meta.get("fq_name", "").split(".")[-1]
    search_col = (meta.get("search_column") or "chunk")
    columns_available: List[str] = meta.get("columns_available", []) or []

    request_columns = intersect_preserving_order_caseaware(columns, columns_available)
    allowed_map = {a.lower(): a for a in columns_available}
    if search_col.lower() not in [c.lower() for c in request_columns]:
        request_columns = [allowed_map.get(search_col.lower(), search_col)] + request_columns

    svc = root.databases[db].schemas[schema].cortex_search_services[short_name]

    doc = svc.search(
        query,
        columns=request_columns,
        filter=filter,
        limit=st.session_state.num_retrieved_chunks,
    )
    results = doc.results  # List[Dict[str, Any]]

    context_rows = []
    lines = []
    search_col_l = search_col.lower()
    search_col_u = search_col.upper()
    for i, r in enumerate(results, start=1):
        content = r.get(search_col) or r.get(search_col_l) or r.get(search_col_u) or ""
        context_rows.append({
            "idx": i,
            "relative_path": r.get("relative_path") or r.get("RELATIVE_PATH"),
            "file_url": r.get("file_url") or r.get("FILE_URL"),
            "chunk": content,
            "score": r.get("score") or r.get("SCORE"),
        })
        lines.append(f"Context document {i}: {content}\n")

    context_text = "".join(lines)
    return context_text, results, context_rows


def build_history_text(chat_history: List[Dict[str, Any]], k: int) -> str:
    if k <= 0 or not chat_history:
        return ""
    turns = chat_history[-k:]
    messages = []
    for t in turns:
        messages.append(f"User: {t.get('question','')}")
        messages.append(f"Assistant: {t.get('answer','')}")
    return "\n".join(messages)


def build_prompt(history_text: str, context_text: str, user_query: str) -> str:
    system = (
        "あなたは正確で簡潔な日本語のアシスタントです。"
        "以下の会話履歴とコンテキストだけを根拠に回答してください。"
        "不明点は推測せず「資料からは断定できません」と述べてください。"
    )
    instructions = (
        "手順:\n"
        "1) コンテキストの要点を把握\n"
        "2) 質問に簡潔に回答（箇条書き歓迎）\n"
        "3) 必要なら注意点や前提条件も記載\n"
    )
    parts = [
        f"System:\n{system}",
        f"Instructions:\n{instructions}",
        f"Conversation history:\n{history_text or '(なし)'}",
        f"Context:\n{context_text or '(該当なし)'}",
        f"User question:\n{user_query}",
    ]
    return "\n\n".join(parts)


# ---------- 表示（チャットUI） ----------

def render_context_table_md(context_rows: List[Dict[str, Any]]) -> str:
    if not context_rows:
        return ""
    md = "| # | PDF | URL | チャンク抜粋 |\n|---:|---|---|---|\n"
    for r in context_rows:
        chunk = r.get("chunk") or ""
        head = (chunk[:160] + "…") if len(chunk) > 160 else chunk
        pdf_disp = (r.get("relative_path") or "").replace("|", "\\|")
        url_disp = (r.get("file_url") or "—").replace("|", "\\|")
        head_disp = head.replace("|", "\\|")
        md += f'| {r.get("idx","")} | {pdf_disp} | {url_disp} | {head_disp} |\n'
    return md


def stream_write_text(container, full_text: str):
    # 疑似ストリーミング（段階描画）
    buf = ""
    step = 80
    for i in range(0, len(full_text), step):
        buf += full_text[i:i+step]
        container.markdown(buf)
        time.sleep(0.02)


def render_existing_history():
    # これまでの履歴をチャットバブルで描画
    for turn in st.session_state.get("chat_history", []):
        with st.chat_message("user"):
            st.markdown(turn.get("question", ""))

        with st.chat_message("assistant"):
            st.markdown(turn.get("answer", ""))

            ctx = turn.get("contexts") or []
            if ctx:
                with st.expander("参照コンテキスト（PDF/URL/チャンク抜粋）", expanded=False):
                    st.markdown(render_context_table_md(ctx))


# ---------- メイン ----------

def main():
    st.title("Cortex Search RAG チャット")

    init_sidebar()
    meta = st.session_state.get("selected_cortex_meta")
    if not meta:
        st.info("左のサイドバーでサービスを選択してください。")
        return

    # 履歴を先に描画
    render_existing_history()

    # 入力（チャット入力）
    user_query = st.chat_input("メッセージを入力")

    if user_query:
        # 直ちにユーザーメッセージを表示
        with st.chat_message("user"):
            st.markdown(user_query)

        # フィルタ最終値の決定
        filter_obj: Dict[str, Any] = {}
        if st.session_state.get("filter_enabled") and st.session_state.get("filter_column"):
            key = st.session_state.get("filter_value_selected_key") or ""
            if key == MANUAL_SENTINEL:
                eff = st.session_state.get("filter_value_manual") or ""
            else:
                eff = key
            if eff:
                filter_obj = {"@eq": {st.session_state["filter_column"]: eff}}

        # 1) 検索 → 文脈生成
        context_text, _results, context_rows = query_cortex_search_service(
            user_query,
            columns=["chunk", "file_url", "relative_path"],
            filter=filter_obj,
        )

        # 2) 履歴ウィンドウ
        history_text = build_history_text(st.session_state.get("chat_history", []), st.session_state.history_k)

        # 3) プロンプト生成
        prompt = build_prompt(history_text, context_text, user_query)

        # 4) LLM実行＋アシスタント表示
        model = st.session_state.selected_model
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                final_text = Complete(model=model, prompt=prompt)
            except Exception as e:
                final_text = f"モデル呼び出しでエラーが発生しました: {e}"

            stream_write_text(placeholder, final_text)

            # 参照チャンクを折りたたみで表示
            if context_rows:
                with st.expander("全文チャンクを見る（回答に渡したテキスト）"):
                    for r in context_rows:
                        st.markdown(f"**#{r['idx']} – {r['relative_path']}**")
                        st.write(r["chunk"])
                        if r["file_url"]:
                            st.write(r["file_url"])
                        st.divider()

        # 5) 履歴へ保存（ターン単位）
        turn = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": user_query,
            "answer": final_text,
            "model": model,
            "contexts": context_rows,
        }
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append(turn)


if __name__ == "__main__":
    main()