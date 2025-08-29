CREATE OR REPLACE DATABASE SNOWFLAKE_INTELLIGENCE;
CREATE OR REPLACE SCHEMA AGENTS;

CREATE OR REPLACE DATABASE handson;
CREATE OR REPLACE SCHEMA handson.car_insurance;
CREATE OR REPLACE STAGE car_insurance_stage
  DIRECTORY = ( ENABLE = TRUE )
  ENCRYPTION = ( TYPE = 'SNOWFLAKE_SSE' )
  COMMENT = '自動車保険データのロード用内部ステージ（ディレクトリテーブル、サーバーサイド暗号化有効）';

CREATE OR REPLACE WAREHOUSE handson_wh
  WAREHOUSE_SIZE = 'X-SMALL';

ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';


--　サンプルデータの作成
/***
POLICIES（契約）…100件
REPAIR_SHOPS（修理工場）…60件
CLAIMS（クレーム）…約140件（1契約あたり0〜3件）
REPAIR_ORDERS（修理明細）…約280行（クレームの一部に明細）

保険料は「基礎料率 × 地域/年齢/車種/用途係数 × 等級(無事故)割引」を計算し、
請求は「事故→報告→査定(見積/認定)→支払/未払残(Reserve)→精算/クローズ」という流れを想定
***/

-- =========================================================
-- 1) REPAIR_SHOPS（60件）
-- =========================================================
CREATE OR REPLACE TABLE REPAIR_SHOPS AS
WITH g AS (
  SELECT SEQ4() AS n
  FROM TABLE(GENERATOR(ROWCOUNT=>100))
),
p AS (
  SELECT
      600000 + n                                            AS SHOP_ID
    , 'AutoCare ' || LPAD(TO_VARCHAR(n+1), 3, '0')          AS SHOP_NAME
    , ARRAY_CONSTRUCT('Tokyo','Kanagawa','Chiba','Saitama','Osaka','Aichi','Hokkaido','Fukuoka','Hyogo','Shizuoka') AS PREFS
    , ARRAY_CONSTRUCT('Chiyoda','Shinjuku','Yokohama','Kawasaki','Funabashi','Urawa','Sapporo','Hakata','Kobe','Shizuoka','Nagoya','Naniwa') AS CITIES
    , ARRAY_CONSTRUCT('Gold','Silver','Bronze')             AS CERTS
    , ARRAY_CONSTRUCT('A','B','C')                          AS TIERS
    , UNIFORM(0,1000,RANDOM())                              AS r1
    , UNIFORM(0,1000,RANDOM())                              AS r2
    , UNIFORM(0,1000,RANDOM())                              AS r3
    , UNIFORM(0,1000,RANDOM())                              AS r4
    , UNIFORM(8000, 14000, RANDOM())                        AS HOURLY_RATE
    , UNIFORM(20, 80, RANDOM())                             AS CAPACITY_PER_MONTH
    , UNIFORM(35, 50, RANDOM()) / 10.0                      AS RATING
    , UNIFORM(5, 12, RANDOM())                              AS TARGET_SLA_DAYS
  FROM g
),
picked AS (
  SELECT
      SHOP_ID,
      SHOP_NAME,
      PREFS,
      CITIES,
      CERTS,
      TIERS,
      (r1 % ARRAY_SIZE(PREFS))  AS PREF_IDX,
      (r2 % ARRAY_SIZE(CITIES)) AS CITY_IDX,
      (r3 % ARRAY_SIZE(CERTS))  AS CERT_IDX,
      (r4 % ARRAY_SIZE(TIERS))  AS TIER_IDX,
      HOURLY_RATE,
      CAPACITY_PER_MONTH,
      RATING,
      TARGET_SLA_DAYS
  FROM p
)
SELECT
    SHOP_ID
  , SHOP_NAME
  , (PREFS[PREF_IDX])::STRING          AS PREFECTURE
  , (CITIES[CITY_IDX])::STRING         AS CITY
  , (CERTS[CERT_IDX])::STRING          AS CERTIFICATION_LEVEL
  , (TIERS[TIER_IDX])::STRING          AS PARTNER_TIER
  , ROUND(HOURLY_RATE,0)               AS HOURLY_RATE_YEN
  , CAPACITY_PER_MONTH
  , RATING
  , TARGET_SLA_DAYS
FROM picked
QUALIFY ROW_NUMBER() OVER (ORDER BY SHOP_ID) <= 60
;

-- =========================================================
-- 2) POLICIES（100件）
-- =========================================================
CREATE OR REPLACE TABLE POLICIES AS
WITH g AS (
  SELECT SEQ4() AS n
  FROM TABLE(GENERATOR(ROWCOUNT=>100))
),
b AS (
  SELECT
      100000 + n                                            AS POLICY_ID
    , 'CUS' || LPAD(TO_VARCHAR(1 + (n % 300)), 5, '0')      AS CUSTOMER_ID
    , DATEADD('day', UNIFORM(0, 700, RANDOM()), TO_DATE('2023-01-01')) AS POLICY_START_DATE
    , ARRAY_CONSTRUCT('Toyota','Nissan','Honda','Mazda','Subaru','Suzuki','Mitsubishi','Lexus','BMW','Mercedes') AS MAKES
    , ARRAY_CONSTRUCT('Prius','Aqua','Fit','Civic','CX-5','Swift','Outlander','RX','3 Series','C-Class') AS MODELS
    , ARRAY_CONSTRUCT('Personal','Commute','Commercial')    AS USES
    , ARRAY_CONSTRUCT('Tokyo','Kanagawa','Chiba','Saitama','Osaka','Aichi','Hokkaido','Fukuoka','Hyogo','Shizuoka') AS PREFS
    , UNIFORM(0,1000,RANDOM()) AS r_mk
    , UNIFORM(0,1000,RANDOM()) AS r_md
    , UNIFORM(0,1000,RANDOM()) AS r_use
    , UNIFORM(0,1000,RANDOM()) AS r_pf
    , UNIFORM(2005, 2024, RANDOM()) AS VEHICLE_YEAR
    , UNIFORM(18, 75, RANDOM()) AS DRIVER_AGE
    , UNIFORM(1, 40, RANDOM()) AS LICENSE_YEARS
    , GREATEST(0, LEAST(10, UNIFORM(0, 12, RANDOM()))) AS NCB_YEARS
    , UNIFORM(0,3,RANDOM()) AS PRIOR_CLAIMS_3Y
    , 45000 AS BASE_RATE_YEN
  FROM g
),
p AS (
  SELECT
      POLICY_ID, CUSTOMER_ID, POLICY_START_DATE,
      DATEADD('day', 365, POLICY_START_DATE) AS POLICY_END_DATE,
      (MAKES[r_mk % ARRAY_SIZE(MAKES)])::STRING  AS VEHICLE_MAKE,
      (MODELS[r_md % ARRAY_SIZE(MODELS)])::STRING AS VEHICLE_MODEL,
      VEHICLE_YEAR,
      (USES[r_use % ARRAY_SIZE(USES)])::STRING   AS VEHICLE_USE,
      DRIVER_AGE, LICENSE_YEARS, NCB_YEARS, PRIOR_CLAIMS_3Y,
      (PREFS[r_pf % ARRAY_SIZE(PREFS)])::STRING  AS REGISTERED_PREF,
      BASE_RATE_YEN
  FROM b
),
rated AS (
  SELECT
      *
    , CASE REGISTERED_PREF WHEN 'Tokyo' THEN 1.20 WHEN 'Kanagawa' THEN 1.15 WHEN 'Osaka' THEN 1.15
                           WHEN 'Hokkaido' THEN 1.10 ELSE 1.00 END AS REGION_FACTOR
    , CASE WHEN DRIVER_AGE < 25 THEN 1.35 WHEN DRIVER_AGE < 30 THEN 1.20
           WHEN DRIVER_AGE < 60 THEN 1.00 ELSE 1.15 END AS AGE_FACTOR
    , CASE WHEN VEHICLE_MAKE IN ('BMW','Mercedes','Lexus') THEN 1.20
           WHEN VEHICLE_MAKE IN ('Mazda','Subaru') THEN 1.05 ELSE 1.00 END AS VEHICLE_FACTOR
    , CASE VEHICLE_USE WHEN 'Commercial' THEN 1.25 WHEN 'Commute' THEN 1.10 ELSE 1.00 END AS USE_FACTOR
    , CASE WHEN PRIOR_CLAIMS_3Y >= 2 THEN 1.30 WHEN PRIOR_CLAIMS_3Y = 1 THEN 1.10 ELSE 1.00 END AS CLAIM_HIST_FACTOR
    , LEAST(0.30, NCB_YEARS * 0.03) AS NCB_DISCOUNT_RATE
    , UNIFORM(30000, 100000, RANDOM()) AS DED_COLLISION_YEN
    , UNIFORM(30000,  80000, RANDOM()) AS DED_COMPREHENSIVE_YEN
  FROM p
),
f AS (
  SELECT
      *
    , ROUND(
        BASE_RATE_YEN
        * REGION_FACTOR * AGE_FACTOR * VEHICLE_FACTOR * USE_FACTOR * CLAIM_HIST_FACTOR
        * (1 - NCB_DISCOUNT_RATE)
      ) AS ANNUAL_PREMIUM_YEN
    , CASE
        WHEN MOD(POLICY_ID,11)=0 THEN 'Lapsed'
        WHEN MOD(POLICY_ID,17)=0 THEN 'Cancelled'
        ELSE 'Active'
      END AS POLICY_STATUS
    , CASE WHEN VEHICLE_YEAR >= 2020 THEN 'Newer'
           WHEN VEHICLE_YEAR >= 2012 THEN 'Mid' ELSE 'Older' END AS VEHICLE_AGE_BAND
  FROM rated
)
SELECT
    POLICY_ID, CUSTOMER_ID, POLICY_START_DATE, POLICY_END_DATE, POLICY_STATUS,
    VEHICLE_MAKE, VEHICLE_MODEL, VEHICLE_YEAR, VEHICLE_AGE_BAND, VEHICLE_USE,
    DRIVER_AGE, LICENSE_YEARS, NCB_YEARS, PRIOR_CLAIMS_3Y, REGISTERED_PREF,
    ROUND(DED_COLLISION_YEN,0) AS DEDUCTIBLE_COLLISION_YEN,
    ROUND(DED_COMPREHENSIVE_YEN,0) AS DEDUCTIBLE_COMP_YEN,
    ANNUAL_PREMIUM_YEN
FROM f
;


-- =========================================================
-- 3) CLAIMS（~140件）―― 必ずREPAIR_SHOPに割当
-- =========================================================
CREATE OR REPLACE TABLE CLAIMS AS
WITH per_policy AS (
  SELECT
      POLICY_ID, CUSTOMER_ID, REGISTERED_PREF,
      DEDUCTIBLE_COLLISION_YEN, DEDUCTIBLE_COMP_YEN,
      -- 1〜3件を整数で安定決定（ハッシュで偏りなし）
      1 + MOD(ABS(HASH(POLICY_ID)), 3) AS CLAIM_CNT
  FROM POLICIES
),
expanded AS (
  -- CLAIM_CNT 件 分を 1..CLAIM_CNT の整数レンジで展開
  SELECT
      p.POLICY_ID, p.CUSTOMER_ID, p.REGISTERED_PREF,
      p.DEDUCTIBLE_COLLISION_YEN, p.DEDUCTIBLE_COMP_YEN,
      seq AS seq
  FROM per_policy p,
       LATERAL FLATTEN(INPUT => ARRAY_GENERATE_RANGE(1, p.CLAIM_CNT + 1))
),
mk AS (
  SELECT
      500000 + ROW_NUMBER() OVER (ORDER BY POLICY_ID, seq) AS CLAIM_ID,
      POLICY_ID, CUSTOMER_ID, REGISTERED_PREF,
      DATEADD('day', UNIFORM(0, 700, RANDOM()), TO_DATE('2023-04-01')) AS INCIDENT_DATE,
      UNIFORM(0,7,RANDOM())  AS REP_DELAY_DAYS,
      UNIFORM(0,30,RANDOM()) AS ASSESS_DELAY_DAYS,
      UNIFORM(1,90,RANDOM()) AS REPAIR_DAYS,
      DEDUCTIBLE_COLLISION_YEN, DEDUCTIBLE_COMP_YEN,
      UNIFORM(0,1000,RANDOM()) AS r1,
      UNIFORM(0,1000,RANDOM()) AS r2
  FROM expanded
),
cause AS (
  SELECT
      *,
      CASE
        WHEN MONTH(INCIDENT_DATE) BETWEEN 7 AND 10 THEN
          CASE WHEN r1 < 380 THEN 'Collision'
               WHEN r1 < 460 THEN 'Weather'
               WHEN r1 < 520 THEN 'Flood'
               WHEN r1 < 610 THEN 'Glass'
               WHEN r1 < 700 THEN 'Vandalism'
               WHEN r1 < 820 THEN 'Theft'
               WHEN r1 < 900 THEN 'Animal'
               ELSE 'Fire' END
        ELSE
          CASE WHEN r1 < 450 THEN 'Collision'
               WHEN r1 < 510 THEN 'Glass'
               WHEN r1 < 590 THEN 'Vandalism'
               WHEN r1 < 700 THEN 'Theft'
               WHEN r1 < 820 THEN 'Weather'
               WHEN r1 < 900 THEN 'Animal'
               ELSE 'Fire' END
      END AS LOSS_CAUSE,
      CASE WHEN r2 < 500 THEN 'Minor'
           WHEN r2 < 800 THEN 'Moderate'
           WHEN r2 < 950 THEN 'Severe'
           ELSE 'Total Loss' END AS SEVERITY
  FROM mk
),
priced AS (
  SELECT
      *,
      CASE
        WHEN LOSS_CAUSE = 'Glass' THEN UNIFORM(40000, 180000, RANDOM())
        WHEN LOSS_CAUSE IN ('Theft','Vandalism') THEN UNIFORM(150000, 600000, RANDOM())
        WHEN LOSS_CAUSE IN ('Weather','Flood') AND SEVERITY IN ('Severe','Total Loss')
             THEN UNIFORM(600000, 2200000, RANDOM())
        WHEN SEVERITY = 'Minor' THEN UNIFORM(30000, 150000, RANDOM())
        WHEN SEVERITY = 'Moderate' THEN UNIFORM(150000, 450000, RANDOM())
        WHEN SEVERITY = 'Severe' THEN UNIFORM(450000, 1000000, RANDOM())
        ELSE UNIFORM(1000000, 2500000, RANDOM())
      END AS ESTIMATE_COST_YEN
  FROM cause
),
dates AS (
  SELECT
      *,
      DATEADD('day', REP_DELAY_DAYS, INCIDENT_DATE) AS REPORTED_DATE,
      DATEADD('day', ASSESS_DELAY_DAYS, DATEADD('day', REP_DELAY_DAYS, INCIDENT_DATE)) AS ASSESSED_DATE,
      DATEADD('day', REPAIR_DAYS, ASSESSED_DATE) AS TARGET_COMPLETE_DATE
  FROM priced
),
money AS (
  SELECT
      *,
      CASE WHEN LOSS_CAUSE IN ('Collision','Vandalism')
           THEN DEDUCTIBLE_COLLISION_YEN ELSE DEDUCTIBLE_COMP_YEN END AS DED_APPLIED,
      UNIFORM(0,100,RANDOM()) AS status_pick
  FROM dates
),
calc AS (
  SELECT
      *,
      ROUND(GREATEST(ESTIMATE_COST_YEN, DED_APPLIED + 10000),0) AS APPROVED_COST_YEN,
      CASE
        WHEN status_pick < 8  THEN 'Denied'
        WHEN status_pick < 30 THEN 'Open'
        WHEN status_pick < 85 THEN 'Approved'
        ELSE 'Paid' END AS CLAIM_STATUS,
      ROUND(UNIFORM(0,100,RANDOM())/100.0, 2) AS FRAUD_SCORE,
      (UNIFORM(0,100,RANDOM()) < 35) AS TOWING_REQUIRED
  FROM money
),
-- 都道府県一致を優先し、1件に1工場をランダムで割り当て
shop_candidates AS (
  SELECT
      c.*,
      s.SHOP_ID,
      IFF(s.PREFECTURE = c.REGISTERED_PREF, 1, 0) AS PREF_MATCH
  FROM calc c
  JOIN REPAIR_SHOPS s ON TRUE
),
ranked AS (
  SELECT
      *,
      ROW_NUMBER() OVER (
        PARTITION BY CLAIM_ID
        ORDER BY PREF_MATCH DESC, HASH(CLAIM_ID, SHOP_ID)
      ) AS rn
  FROM shop_candidates
),
picked AS (
  SELECT *
  FROM ranked
  QUALIFY rn = 1
),
limited AS (
  -- ここで全体 140 件以内に整形（最低100件は確実に確保されます）
  SELECT *, ROW_NUMBER() OVER (ORDER BY POLICY_ID, INCIDENT_DATE) AS rn_global
  FROM picked
)
SELECT
    CLAIM_ID,
    POLICY_ID,
    CUSTOMER_ID,
    INCIDENT_DATE,
    REPORTED_DATE,
    ASSESSED_DATE,
    TARGET_COMPLETE_DATE,
    REGISTERED_PREF                       AS INCIDENT_PREFECTURE,
    LOSS_CAUSE,
    SEVERITY,
    ROUND(ESTIMATE_COST_YEN,0)            AS ESTIMATE_COST_YEN,
    DED_APPLIED                            AS DEDUCTIBLE_APPLIED_YEN,
    ROUND(APPROVED_COST_YEN,0)            AS APPROVED_COST_YEN,
    CASE
      WHEN CLAIM_STATUS='Denied'   THEN 0
      WHEN CLAIM_STATUS='Open'     THEN ROUND(APPROVED_COST_YEN * 0.4,0)
      WHEN CLAIM_STATUS='Approved' THEN ROUND(APPROVED_COST_YEN * 0.8,0)
      ELSE APPROVED_COST_YEN
    END                                   AS PAID_TO_DATE_YEN,
    ROUND(GREATEST(APPROVED_COST_YEN - (
      CASE
        WHEN CLAIM_STATUS='Denied'   THEN 0
        WHEN CLAIM_STATUS='Open'     THEN ROUND(APPROVED_COST_YEN * 0.4,0)
        WHEN CLAIM_STATUS='Approved' THEN ROUND(APPROVED_COST_YEN * 0.8,0)
        ELSE APPROVED_COST_YEN
      END
    ),0),0)                               AS OUTSTANDING_RESERVE_YEN,
    CLAIM_STATUS,
    FRAUD_SCORE,
    TOWING_REQUIRED,
    SHOP_ID                                AS REPAIR_SHOP_ID
FROM limited
WHERE rn_global <= 140
;



-- =========================================================
-- 4) REPAIR_ORDERS（~280行）―― 明細は必ず存在
-- =========================================================
CREATE OR REPLACE TABLE REPAIR_ORDERS AS
WITH target_claims AS (
  SELECT c.*, s.HOURLY_RATE_YEN
  FROM CLAIMS c
  JOIN REPAIR_SHOPS s ON c.REPAIR_SHOP_ID = s.SHOP_ID       -- INNER JOINでNULL排除
  WHERE c.CLAIM_STATUS IN ('Approved','Paid','Open')
),
expanded AS (
  SELECT
      c.CLAIM_ID, c.REPAIR_SHOP_ID, c.ASSESSED_DATE, c.TARGET_COMPLETE_DATE,
      c.APPROVED_COST_YEN, c.CLAIM_STATUS, c.HOURLY_RATE_YEN,
      1 + (UNIFORM(0,3,RANDOM())) AS LINE_CNT
  FROM target_claims c
),
lines AS (
  SELECT
      CLAIM_ID, REPAIR_SHOP_ID, ASSESSED_DATE, TARGET_COMPLETE_DATE,
      APPROVED_COST_YEN, CLAIM_STATUS, HOURLY_RATE_YEN,
      ROW_NUMBER() OVER (PARTITION BY CLAIM_ID ORDER BY SEQ4()) AS LINE_NO
  FROM expanded, LATERAL FLATTEN(INPUT => ARRAY_GENERATE_RANGE(0, LINE_CNT))
),
priced AS (
  SELECT
      *
    , UNIFORM(5, 20, RANDOM())              AS LABOR_HOURS
    , UNIFORM(20000, 150000, RANDOM())      AS PARTS_COST_YEN
  FROM lines
),
calc AS (
  SELECT
      *
    , ROUND(LABOR_HOURS * HOURLY_RATE_YEN,0)             AS LABOR_COST_YEN
    , ROUND(PARTS_COST_YEN + (LABOR_HOURS * HOURLY_RATE_YEN),0) AS SUBTOTAL_YEN
    , ROUND(SUBTOTAL_YEN * 0.10,0)                       AS TAX_YEN
    , ROUND(SUBTOTAL_YEN + (SUBTOTAL_YEN * 0.10),0)      AS LINE_TOTAL_YEN
    , DATEADD('day', UNIFORM(0, 3, RANDOM()), ASSESSED_DATE) AS WORK_START_DATE
    , DATEADD('day', UNIFORM(3, 20, RANDOM()), ASSESSED_DATE) AS WORK_END_DATE
  FROM priced
),
cap AS (
  SELECT
      *
    , SUM(LINE_TOTAL_YEN) OVER (PARTITION BY CLAIM_ID) AS CLAIM_SUM
    , CASE
        WHEN CLAIM_STATUS='Open'     THEN APPROVED_COST_YEN * 0.90
        WHEN CLAIM_STATUS='Approved' THEN APPROVED_COST_YEN * 0.95
        ELSE APPROVED_COST_YEN * 1.00
      END AS TARGET_SUM
  FROM calc
),
scaled AS (
  SELECT
      *
    , CASE WHEN CLAIM_SUM = 0 THEN LINE_TOTAL_YEN
           ELSE ROUND(LINE_TOTAL_YEN * (TARGET_SUM / CLAIM_SUM), 0)
      END AS ADJUSTED_LINE_TOTAL_YEN
  FROM cap
)
SELECT
    700000 + ROW_NUMBER() OVER (ORDER BY CLAIM_ID, LINE_NO) AS ORDER_ID
  , CLAIM_ID
  , REPAIR_SHOP_ID
  , LINE_NO
  , ROUND(PARTS_COST_YEN,0) AS PARTS_COST_YEN
  , ROUND(LABOR_HOURS,1)    AS LABOR_HOURS
  , ROUND(HOURLY_RATE_YEN,0) AS LABOR_RATE_YEN
  , ROUND(LABOR_COST_YEN,0) AS LABOR_COST_YEN_BEFORE_ADJ
  , ROUND(ADJUSTED_LINE_TOTAL_YEN/1.10,0) AS SUBTOTAL_AFTER_ADJ_YEN
  , ROUND(ADJUSTED_LINE_TOTAL_YEN - (ADJUSTED_LINE_TOTAL_YEN/1.10),0) AS TAX_AFTER_ADJ_YEN
  , ADJUSTED_LINE_TOTAL_YEN AS LINE_TOTAL_AFTER_ADJ_YEN
  , WORK_START_DATE
  , WORK_END_DATE
FROM scaled
QUALIFY ROW_NUMBER() OVER (ORDER BY CLAIM_ID, LINE_NO) <= 280
;

-- Git連携のため、API統合を作成する
CREATE OR REPLACE API INTEGRATION git_api_integration
  API_PROVIDER = git_https_api
  API_ALLOWED_PREFIXES = ('https://github.com/sf-yitagaki/')
  ENABLED = TRUE;

-- GIT統合の作成
CREATE OR REPLACE GIT REPOSITORY GIT_INTEGRATION_FOR_HANDSON
  API_INTEGRATION = git_api_integration
  ORIGIN = 'git@github.com:sf-yitagaki/car_insrurance_handson.git';

-- チェックする
ls @GIT_INTEGRATION_FOR_HANDSON/branches/main;

-- Githubからファイルを持ってくる
COPY FILES INTO @handson.car_insurance.car_insurance_stage FROM @GIT_INTEGRATION_FOR_HANDSON/branches/main/data/ PATTERN = '.*insurance_claim_form\\.pdf$';

COPY FILES INTO @handson.car_insurance.terms_and_conditions FROM @GIT_INTEGRATION_FOR_HANDSON/branches/main/data/ PATTERN = '.*yakkan_driver_insurance\\.pdf$';


// Step4: NotebookとStreamlitを作成 //

-- Notebookの作成
CREATE OR REPLACE NOTEBOOK cortex_handson_part1
    FROM @GIT_INTEGRATION_FOR_HANDSON/branches/main/handson1
    MAIN_FILE = 'car_insurance_analysis.ipynb'
    QUERY_WAREHOUSE = handson_WH
    WAREHOUSE = handson_WH;