# 專案目標（GraphMER 從醫學轉移到編碼/軟體工程領域）

本文件彙整您提出的目標與關鍵要求，作為專案規格與實作的依據。

## 目標概述
- 將 GraphMER 的神經符號（Neurosymbolic）架構從醫學領域轉移至編碼/軟體工程領域。
- 保留高效的 Encoder-only Transformer（約 80M 參數，BERT/Roberta 系列）與關鍵方法（Leafy Chain Graph Encoding、HGAT、雙重訓練目標）。
- 以編碼領域的本體論（Ontology）與知識圖譜（KG）取代醫學領域知識，構建可靠、可對齊語義的編碼知識圖譜與語義理解能力。

## 必要更動（從醫學 → 編碼）
1. 本體論（Ontology）
   - 建立/採用編碼領域的正式本體論，涵蓋類別、函數、型別、模組、API、依賴等；
   - 定義關係（例如：inherits_from、calls_function、depends_on、implements、overrides、imports、defines、declares 等）及其約束（方向、型別、基數、多語言差異）。

2. 種子知識圖譜（Seed KG）
   - 以專家策劃的小型高品質三元組（例如 ~28k triples）作為初始語義與本體論對齊範例；
   - 附帶來源、語言、repo、commit 與程式碼 span 以溯源與審計。

3. 語料來源（Syntactic Source / 文字來源）
   - 大規模程式碼與技術文件：開源程式碼庫、API 文件、README、架構說明、型別 stub 等；
   - 覆蓋 1–2 種語言起步（建議 Python + Java）以利泛化與異質驗證。

4. 分詞器（Tokenizer）
   - 針對編碼詞彙重新訓練（BPE/Unigram），需懂 camelCase/snake_case、保留語法標點（(), {}, :, ., ->）；
   - 為關係/圖形分隔保留特殊 token；避免過度切分或語義碰撞。

## 訓練與建模要求
- Leafy Chain Graph Encoding：將句法空間（原始代碼/文檔 token）與語義空間（KG 尾部實體）扁平化為單一序列；
  - 根節點：來自代碼/文檔的 token 與上下文；
  - 葉節點：附加的 KG 尾部實體 token（帶關係型別）。
- HGAT（分層圖注意力）：在嵌入層啟用關係感知注意力，強化本體論對齊；
- 雙重訓練目標：
  - MLM（Mask Language Modeling）學習語法與上下文；
  - MNM（Mask Node Modeling）學習 KG 語義與事實；
  - 加上型別一致負採樣、約束正則（如反對稱/傳遞性約束）。

## 輔助任務
- 實體發現：透過輔助 LLM 或解析工具抽取類別/函數/型別等實體；
- 關係選擇：決定輸入 GraphMER 的三元組關係子集；
- 結果組合：將 token 級輸出組裝為完整尾部實體（如函數簽名）。

## 預期成果
- 一個以編碼本體為核心、能遵循語義規則（避免統計幻覺）的高效 Encoder 模型；
- 覆蓋 call-graph、依賴推斷、型別/名稱消歧、程式碼搜尋重排序等任務的知識圖譜與模型能力；
- 可擴展到多語言與多框架的通用編碼知識層。

## 計算資源與團隊（Compute and Team）
- 訓練資源：A100 (40–80GB) 或 A10 (24GB)；M2 冒煙/消融 2–4 張，M3 正式訓練 8 張；儲存 2–4 TB。
- 伺服：staging 1 個 GPU 節點；production POC 2–3 節點自動擴縮。
- 團隊配置：KG/本體 1、解析/靜態分析 2、ML 1–2、MLOps 1、技術 PM 0.5。

## 量化成功指標（Quantitative Success Metrics）
- KG 連結預測：MRR ≥ 0.52、Hits@10 ≥ 0.78；本體違規 ≤ 1%。
- 消歧：top-1 準確率 ≥ 92%。
- 程式碼搜尋重排序：MRR@10 ≥ 0.44，對 CodeBERT 基線相對提升 ≥ 10%。
- 呼叫圖補全：F1 ≥ 0.63，較靜態分析基線 +8–12%。
- 依賴推斷：F1 ≥ 0.70，較啟發式 +10%。
- 延遲：P50 ≤ 25 ms、P95 ≤ 60 ms（512 tokens）。

## 種子 KG 範圍說明（Seed KG Scope）
- 初始種子 20–50k 高品質三元組作為穩定訓練與本體錨點（非硬性 28k 限定）。
- 後續擴展至 0.5–2M 自動抽取三元組，配合信心評分與主動錯誤挖掘提升品質。

## 風險摘要（Risk Summary）
- 本體漂移、KG 噪聲、分詞器切分錯誤、負採樣捷徑、授權限制；
- 緩解：CI schema gate、靜態分析結合、identifier-aware tokenizer、型別一致/困難負採樣、授權掃描與白名單策略。

---

# Project Objective (EN)

Summarizes your target for adapting GraphMER to the software engineering domain.

## Objective
- Adapt GraphMER’s neurosymbolic encoder from the biomedical domain to software engineering while keeping the efficient ~80M encoder-only Transformer and key mechanisms (Leafy Chain Graph Encoding, HGAT, dual MLM+MNM objectives).
- Replace medical ontology/KG with a formal software code ontology and graph, enabling ontology-aligned semantic reasoning over code and technical docs.

## Required Domain Replacements
- Ontology: typed entities/relations and constraints across code elements (class, function, type, module, API, dependency, etc.).
- Seed KG: curated ~28k high-quality triples with provenance.
- Corpus: large-scale code and docs (repos, API docs, READMEs, ADRs, type stubs), start with 1–2 languages.
- Tokenizer: code-aware BPE/Unigram with identifier rules and relation/special tokens.

## Training & Modeling
- Leafy Chain Graph Encoding, HGAT for relation-aware attention and ontology alignment.
- Dual objectives: MLM on code/doc tokens + MNM on KG leaf tails; type-consistent negatives and ontology-constraint regularizers.

## Auxiliary Tasks
- Entity discovery, relation selection, and output composition for complete tail entities (e.g., function signatures).

## Outcomes
- An ontology-respecting encoder enabling call-graph completion, dependency inference, disambiguation, and code search reranking, extensible across languages/frameworks.
