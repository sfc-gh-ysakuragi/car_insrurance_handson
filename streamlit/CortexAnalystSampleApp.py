# =============================================================================
# アプリ名: Cortex Analyst アプリ（日本語 UI）
# =============================================================================
# 概要:
# - Snowflake Cortex Analyst と連携し、自然言語（日本語）でデータに質問・分析できる
#   Streamlit アプリです。
# - データベース → スキーマ → セマンティックビュー（FQN）を順に選択し、Cortex Analyst
#   API に対話履歴と選択したセマンティックビューを送信して回答（テキスト/SQL/可視化）を得ます。
#
# 主な機能:
# - チャット体験
#   - 初回: セマンティックビュー選択済みなら自動プロンプト送信
#   - 入力: st.chat_input から日本語で質問
#   - 応答: Cortex Analyst API からのメッセージ/SQL/警告を表示
# - SQL の表示と実行
#   - 生成 SQL を開閉表示
#   - Verified Query Repository 情報のポップオーバー表示
#   - Snowpark 経由で SQL 実行 → データタブ/グラフタブ（折れ線/棒）で可視化
# - フィードバック送信
#   - 生成 SQL に対し「👍/👎」と任意コメントを API へ送信
# - エラーハンドリング
#   - API ステータス/メッセージの整形表示
#   - ストリームリットのトースト通知
#
# 前提条件:
# - Snowflake セッションが有効（Snowflake Notebooks / Streamlit 実行環境）
# - 対象セマンティックビューにアクセス可能なロール権限
# - Cortex Analyst API 利用が許可されていること
# - 参照するセマンティックビューが事前に作成済み
#
# 参考:
# - Snowflake Quickstart（セマンティックビュー）
#   https://quickstarts.snowflake.com/guide/snowflake-semantic-view/index.html?index=..%2F..index#6
#
# 設定/定数:
# - API_ENDPOINT / FEEDBACK_API_ENDPOINT / API_TIMEOUT をコード内定数で管理
#
#
# 作成者: Sakuragi (Snowflake)
# 最終更新日: 2025-10-05
# =============================================================================
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import _snowflake
import pandas as pd
import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSQLException

API_ENDPOINT = "/api/v2/cortex/analyst/message"
FEEDBACK_API_ENDPOINT = "/api/v2/cortex/analyst/feedback"
API_TIMEOUT = 50000

session = get_active_session()


def main():
    if "messages" not in st.session_state:
        reset_session_state()
    show_header_and_sidebar()

    # セマンティックビューが選ばれている場合のみ自動メッセージ
    if len(st.session_state.messages) == 0 and st.session_state.get("selected_semantic_view_fqn"):
        process_user_input("どんな分析ができますか？日本語で教えてください。")

    display_conversation()
    handle_user_inputs()
    handle_error_notifications()
    #display_warnings()


def reset_session_state():
    st.session_state.messages = []
    st.session_state.active_suggestion = None
    st.session_state.warnings = []
    st.session_state.form_submitted = {}


@st.cache_data(show_spinner=False)
def list_databases() -> List[str]:
    candidates: List[str] = []
    # 1) 標準: SHOW DATABASES
    try:
        df = session.sql("SHOW DATABASES").to_pandas()
        df.columns = [str(c).lower() for c in df.columns]
        if "name" in df.columns:
            candidates.extend([str(x) for x in df["name"].tolist() if x])
    except Exception:
        pass
    # 2) フェイルオーバー: ACCOUNT_USAGE（権限ある場合のみ）
    try:
        df2 = session.sql(
            "SELECT DATABASE_NAME FROM SNOWFLAKE.ACCOUNT_USAGE.DATABASES "
            "WHERE DELETED IS NULL ORDER BY DATABASE_NAME"
        ).to_pandas()
        df2.columns = [str(c).lower() for c in df2.columns]
        if "database_name" in df2.columns:
            candidates.extend([str(x) for x in df2["database_name"].tolist() if x])
    except Exception:
        pass
    return sorted({c for c in candidates})


def _quote_ident(name: str) -> str:
    s = str(name).replace('"', '""')
    return f'"{s}"'


@st.cache_data(show_spinner=False)
def list_schemas(database: str) -> List[str]:
    if not database:
        return []
    try:
        df = session.sql(
            f"SELECT SCHEMA_NAME FROM {_quote_ident(database)}.INFORMATION_SCHEMA.SCHEMATA ORDER BY SCHEMA_NAME"
        ).to_pandas()
        df.columns = [str(c).lower() for c in df.columns]
        col = "schema_name" if "schema_name" in df.columns else None
        if col is None:
            return []
        return [str(x) for x in df[col].tolist()]
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def list_semantic_views(database: str, schema: str) -> List[str]:
    if not database or not schema:
        return []
    try:
        # クオート付き SHOW（小文字/特殊文字対応）
        q = f"SHOW SEMANTIC VIEWS IN SCHEMA {_quote_ident(database)}.{_quote_ident(schema)}"
        df = session.sql(q).to_pandas()

        # 列名を正規化（前後空白/二重引用符を除去して小文字化）
        norm_map = {orig: str(orig).strip().strip('"').lower() for orig in df.columns}
        # 'name' に対応する元の列名を特定（name, "name", NAME などを吸収）
        names_col = next((orig for orig, norm in norm_map.items() if norm == "name"), None)
        if not names_col:
            return []

        names = [str(x).strip() for x in df[names_col].dropna().tolist() if str(x).strip()]
        return [f"{database}.{schema}.{n}" for n in names]
    except Exception:
        return []


def _on_database_change():
    try:
        list_schemas.clear()
        list_semantic_views.clear()
    except Exception:
        pass
    st.session_state.selected_schema = None
    st.session_state.selected_semantic_view_fqn = None
    reset_session_state()


def _on_schema_change():
    try:
        list_semantic_views.clear()
    except Exception:
        pass
    st.session_state.selected_semantic_view_fqn = None
    reset_session_state()


def _on_view_change():
    reset_session_state()


def _clear_metadata_cache():
    try:
        list_databases.clear()
        list_schemas.clear()
        list_semantic_views.clear()
    except Exception:
        pass


def _get_current_context_defaults() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    try:
        df = session.sql(
            "SELECT CURRENT_DATABASE() AS DB, CURRENT_SCHEMA() AS SCHEMA, CURRENT_WAREHOUSE() AS WH, CURRENT_ROLE() AS ROLE"
        ).to_pandas()
        db = str(df["DB"].iloc[0]) if not pd.isna(df["DB"].iloc[0]) else None
        sc = str(df["SCHEMA"].iloc[0]) if not pd.isna(df["SCHEMA"].iloc[0]) else None
        wh = str(df["WH"].iloc[0]) if not pd.isna(df["WH"].iloc[0]) else None
        role = str(df["ROLE"].iloc[0]) if not pd.isna(df["ROLE"].iloc[0]) else None
        return db, sc, wh, role
    except Exception:
        return None, None, None, None


def show_header_and_sidebar():
    st.title("Cortex Analyst")
    st.markdown("Cortex Analyst へようこそ。下の入力欄に質問を日本語で入力してください。")

    with st.sidebar:
        db_default, schema_default, wh, role = _get_current_context_defaults()

        st.caption("接続情報")
        ctx_cols = st.columns(2)
        ctx_cols[0].text_input("現在のロール", value=role or "", disabled=True)
        ctx_cols[1].text_input("現在のウェアハウス", value=wh or "", disabled=True)

        st.divider()
        st.caption("セマンティックビューの選択")

        # Database
        dbs = list_databases()
        if "selected_database" not in st.session_state or st.session_state.selected_database not in dbs:
            st.session_state.selected_database = db_default if db_default in dbs else (dbs[0] if dbs else None)

        st.selectbox(
            "データベース",
            dbs,
            key="selected_database",
            on_change=_on_database_change,
        )

        # Schema
        schemas = list_schemas(st.session_state.selected_database) if st.session_state.selected_database else []
        if "selected_schema" not in st.session_state or st.session_state.selected_schema not in schemas:
            st.session_state.selected_schema = schema_default if schema_default in schemas else (schemas[0] if schemas else None)

        st.selectbox(
            "スキーマ",
            schemas,
            key="selected_schema",
            on_change=_on_schema_change,
            disabled=not bool(schemas),
        )

        # Semantic View
        views = (
            list_semantic_views(st.session_state.selected_database, st.session_state.selected_schema)
            if st.session_state.selected_database and st.session_state.selected_schema
            else []
        )
        if (
            "selected_semantic_view_fqn" not in st.session_state
            or st.session_state.selected_semantic_view_fqn not in views
        ):
            st.session_state.selected_semantic_view_fqn = views[0] if views else None

        if views:
            st.selectbox(
                "セマンティックビュー",
                views,
                format_func=lambda s: s.split(".")[-1] if s else s,
                key="selected_semantic_view_fqn",
                on_change=_on_view_change,
            )
        else:
            st.info("選択中のデータベース/スキーマにセマンティックビューが見つかりませんでした。")

        st.caption("選択中の FQN")
        st.code(st.session_state.get("selected_semantic_view_fqn") or "(未選択)", language="text")

        st.divider()
        btn_cols = st.columns(2)
        if btn_cols[0].button("🔄 メタデータを更新", use_container_width=True):
            _clear_metadata_cache()
            st.rerun()
        if btn_cols[1].button("会話履歴をクリア", use_container_width=True):
            reset_session_state()
            st.rerun()


def handle_user_inputs():
    if not st.session_state.get("selected_semantic_view_fqn"):
        st.info("左のサイドバーでデータベース・スキーマ・セマンティックビューを選択してください。")
    user_input = st.chat_input("ご質問を入力してください")
    if user_input:
        if not st.session_state.get("selected_semantic_view_fqn"):
            st.toast("セマンティックビューを選択してください。", icon="⚠️")
            return
        process_user_input(user_input)
    elif st.session_state.active_suggestion is not None:
        suggestion = st.session_state.active_suggestion
        st.session_state.active_suggestion = None
        process_user_input(suggestion)


def handle_error_notifications():
    if st.session_state.get("fire_API_error_notify"):
        st.toast("API エラーが発生しました！", icon="🚨")
        st.session_state["fire_API_error_notify"] = False


def process_user_input(prompt: str):
    st.session_state.warnings = []

    new_user_message = {
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    }
    st.session_state.messages.append(new_user_message)
    with st.chat_message("user"):
        user_msg_index = len(st.session_state.messages) - 1
        display_message(new_user_message["content"], user_msg_index)

    with st.chat_message("analyst"):
        with st.spinner("Cortex Analyst の応答を待機中..."):
            time.sleep(1)
            response, error_msg = get_analyst_response(st.session_state.messages)
            if error_msg is None:
                analyst_message = {
                    "role": "analyst",
                    "content": response["message"]["content"],
                    "request_id": response["request_id"],
                }
            else:
                analyst_message = {
                    "role": "analyst",
                    "content": [{"type": "text", "text": error_msg}],
                    "request_id": response["request_id"],
                }
                st.session_state["fire_API_error_notify"] = True

            if "warnings" in response:
                st.session_state.warnings = response["warnings"]

            st.session_state.messages.append(analyst_message)
            st.rerun()


def display_warnings():
    warnings = st.session_state.warnings
    for warning in warnings:
        st.warning(warning["message"], icon="⚠️")


def get_analyst_response(messages: List[Dict]) -> Tuple[Dict, Optional[str]]:
    request_body = {
        "messages": messages,
        "semantic_view": st.session_state.get("selected_semantic_view_fqn"),
    }
    resp = _snowflake.send_snow_api_request(
        "POST",
        API_ENDPOINT,
        {},
        {},
        request_body,
        None,
        API_TIMEOUT,
    )

    parsed_content = json.loads(resp["content"])

    if resp["status"] < 400:
        return parsed_content, None
    else:
        error_msg = f"""
🚨 Cortex Analyst API エラー 🚨

* 応答コード: `{resp['status']}`
* リクエストID: `{parsed_content.get('request_id', '(unknown)')}`
* エラーコード: `{parsed_content.get('error_code', '(unknown)')}`

メッセージ:
        """
        return parsed_content, error_msg


def display_conversation():
    for idx, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            if role == "analyst":
                display_message(content, idx, message.get("request_id"))
            else:
                display_message(content, idx)


def display_message(
    content: List[Dict[str, Union[str, Dict]]],
    message_index: int,
    request_id: Union[str, None] = None,
):
    for item in content:
        if item["type"] == "text":
            st.markdown(item["text"])
        elif item["type"] == "suggestions":
            for suggestion_index, suggestion in enumerate(item["suggestions"]):
                if st.button(
                    suggestion, key=f"suggestion_{message_index}_{suggestion_index}"
                ):
                    st.session_state.active_suggestion = suggestion
        elif item["type"] == "sql":
            display_sql_query(
                item["statement"], message_index, item.get("confidence"), request_id
            )
        else:
            pass


@st.cache_data(show_spinner=False)
def get_query_exec_result(query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    global session
    try:
        df = session.sql(query).to_pandas()
        return df, None
    except SnowparkSQLException as e:
        return None, str(e)


def display_sql_confidence(confidence: dict):
    if confidence is None:
        return
    verified_query_used = confidence.get("verified_query_used")
    with st.popover(
        "検証済みクエリの利用状況",
        help="Verified Query Repository にある検証済みクエリを用いて SQL が生成されたかの情報です（詳細はドキュメントを参照）。",
    ):
        with st.container():
            if verified_query_used is None:
                st.text("この回答の生成に Verified Query は使用されていません。")
                return
            st.text(f"名称: {verified_query_used.get('name')}")
            st.text(f"質問: {verified_query_used.get('question')}")
            st.text(f"検証者: {verified_query_used.get('verified_by')}")
            st.text(f"検証日時: {datetime.fromtimestamp(verified_query_used.get('verified_at'))}")
            st.text("SQL クエリ:")
            st.code(verified_query_used.get("sql", ""), language="sql", wrap_lines=True)


def display_sql_query(
    sql: str, message_index: int, confidence: dict, request_id: Union[str, None] = None
):
    with st.expander("生成された SQL", expanded=False):
        st.code(sql, language="sql")
        display_sql_confidence(confidence)

    with st.expander("結果", expanded=True):
        with st.spinner("SQL を実行しています..."):
            df, err_msg = get_query_exec_result(sql)
            if df is None:
                st.error(f"生成された SQL を実行できませんでした。エラー: {err_msg}")
            elif df.empty:
                st.write("データは返されませんでした。")
            else:
                data_tab, chart_tab = st.tabs(["データ 📄", "グラフ 📉"])
                with data_tab:
                    st.dataframe(df, use_container_width=True)

                with chart_tab:
                    display_charts_tab(df, message_index)
    if request_id:
        display_feedback_section(request_id)


def display_charts_tab(df: pd.DataFrame, message_index: int) -> None:
    if len(df.columns) >= 2:
        all_cols_set = set(df.columns)
        col1, col2 = st.columns(2)
        x_col = col1.selectbox("X 軸", all_cols_set, key=f"x_col_select_{message_index}")
        y_col = col2.selectbox("Y 軸", all_cols_set.difference({x_col}), key=f"y_col_select_{message_index}")
        chart_type = st.selectbox("グラフ種類を選択", options=["折れ線グラフ 📈", "棒グラフ 📊"], key=f"chart_type_{message_index}")
        if chart_type == "折れ線グラフ 📈":
            st.line_chart(df.set_index(x_col)[y_col])
        elif chart_type == "棒グラフ 📊":
            st.bar_chart(df.set_index(x_col)[y_col])
    else:
        st.write("グラフには 2 列以上の列が必要です。")


def display_feedback_section(request_id: str):
    with st.popover("📝 クエリのフィードバック"):
        if request_id not in st.session_state.form_submitted:
            with st.form(f"feedback_form_{request_id}", clear_on_submit=True):
                positive = st.radio("生成された SQL の評価", options=["👍", "👎"], horizontal=True)
                positive = positive == "👍"
                submit_disabled = (
                    request_id in st.session_state.form_submitted
                    and st.session_state.form_submitted[request_id]
                )
                feedback_message = st.text_input("任意のフィードバック")
                submitted = st.form_submit_button("送信", disabled=submit_disabled)
                if submitted:
                    err_msg = submit_feedback(request_id, positive, feedback_message)
                    st.session_state.form_submitted[request_id] = {"error": err_msg}
                    st.session_state.popover_open = False
                    st.rerun()
        elif request_id in st.session_state.form_submitted and st.session_state.form_submitted[request_id]["error"] is None:
            st.success("フィードバックを送信しました。", icon="✅")
        else:
            st.error(st.session_state.form_submitted[request_id]["error"])


def submit_feedback(request_id: str, positive: bool, feedback_message: str) -> Optional[str]:
    request_body = {
        "request_id": request_id,
        "positive": positive,
        "feedback_message": feedback_message,
    }
    resp = _snowflake.send_snow_api_request(
        "POST",
        FEEDBACK_API_ENDPOINT,
        {},
        {},
        request_body,
        None,
        API_TIMEOUT,
    )
    if resp["status"] == 200:
        return None

    parsed_content = json.loads(resp["content"])
    err_msg = f"""
        🚨 Cortex Analyst API エラー 🚨
        
        * 応答コード: `{resp['status']}`
        * リクエストID: `{parsed_content.get('request_id', '(unknown)')}`
        * エラーコード: `{parsed_content.get('error_code', '(unknown)')}`
        
        メッセージ:
        ```
        {parsed_content.get('message', '')}
        ```
        """
    return err_msg


if __name__ == "__main__":
    main()