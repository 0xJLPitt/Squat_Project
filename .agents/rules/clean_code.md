# Clean Code 軟體設計規範與重構準則 (Clean Code Guidelines)

本規範彙整自 Uncle Bob《Clean Code》與 [clean-code-python](https://github.com/zedr/clean-code-python) 的核心哲學，涵蓋 Python 與通用/Dart 專案之核心重構與開發準則。
**所有程式碼編寫、新增、修改與重構均必須嚴格遵循以下原則。**

---

## 1. 變數與命名原則 (Meaningful Names)

### 1.1 意圖明確且可發音 (Intent-Revealing & Pronounceable)
- 變數與函式名稱必須能自我解釋「它是什麼」與「它做什麼」。
- ❌ 避免無意義的縮寫：`int d;`, `var ymd = ...;`, `void calc();`, `df1, df2 = ...`
- ✅ 使用明確命名：`int elapsedTimeInDays;`, `final createdAt = ...;`, `def calculate_total_score():`

### 1.2 同一概念使用相同詞彙 (Consistent Vocabulary)
- 獲取資料統一使用 `get_...` 或 `fetch_...`，不要在同個專案混用 `get_users`, `query_members`, `retrieve_accounts`。

### 1.3 避免心智轉換與多餘上下文 (Avoid Mental Mapping & Redundant Context)
- 不要讓讀者猜測 `i`, `j`, `k`, `data`, `info`, `item` 代表什麼。
- 若類別名為 `Runner`，欄位不應命名為 `runner_name`，使用 `name` 即可（除非為了跨模組避免歧義）。

### 1.4 可搜尋的名稱與消除魔法常數 (Searchable Names & No Magic Values)
- ❌ 避免直接在代碼中使用寫死數字或字串：`if status == 2:`、`delay(86400)`
- ✅ 抽取為命名常數或 Enum：`if status == UploadStatus.COMPLETED:`、`SECONDS_PER_DAY = 86400`

---

## 2. 函式與方法設計 (Functions)

### 2.1 單一職責與只做一件事 (Do One Thing & SRP)
- 每個函式只做一件明確的事情，並且把它做好。
- 單一函式長度建議不超過 20~30 行。

### 2.2 單一抽象層次 (Single Level of Abstraction - SLA)
- 一個函式內的語句應處於同一抽象層級。高階業務流程（如：`validate_input() -> process_video() -> save_results()`）不應與低階細節（如：位元組計算、字串正規化）混在同一個函式內。

### 2.3 嚴格限制參數數量 (Reduce Arguments)
- 最理想的參數數量是 0~2 個。超過 3 個參數時：
  - 改用具名參數 / 關鍵字引數（Keyword Arguments）。
  - 或封裝成 Dataclass / Parameter Object / Config Dict。

### 2.4 禁止使用布林 Flag 作為參數 (Avoid Flag Arguments)
- ❌ 避免：`def process_data(is_separate: bool):`
- ✅ 拆分成兩個獨立函式：`def process_data_unified():` 與 `def process_data_separately():`。

### 2.5 避免副作用 (No Side Effects)
- 函式承諾做的事情就是它唯一做的事情。查詢函式（Query）不應暗中修改全域狀態或內部成員。

### 2.6 提早返回，避免深層巢狀 (Guard Clauses & Return Early)
- ❌ 避免箭頭型的多層 `if-else`。
- ✅ 優先使用 Guard Clauses 檢查邊界條件並提早 `return` 或 `raise`。

---

## 3. 物件與資料結構 (Objects & Data Structures)

### 3.1 優先追求不可變性 (Favor Immutability)
- 實體與狀態模型優先使用不可變資料結構（如 Python 的 `@dataclass(frozen=True)`、NamedTuple，或 Dart 的 `final` 與 `const`）。
- 產生新狀態時盡量使用複製與新實例，避免直接原地突變（In-place Mutation）引發隱蔽副作用。

### 3.2 良好封裝 (Encapsulation & Hide Implementation)
- 內部集合（如 `list`, `dict`）應避免直接暴露給外部隨意修改，透過專用方法或防禦性複製操作。

---

## 4. 類別架構與 SOLID 原則 (Classes & SOLID)

- **S (Single Responsibility)**: 
  - 各模組職責劃分清晰（資料讀取、演算法計算、視覺化渲染分離）。
- **O (Open/Closed)**: 
  - 對擴展開放，對修改封閉（透過介面多型或抽象基類擴充功能，而非狂加 `if-elif-else`）。
- **L (Liskov Substitution)**: 
  - 子類別/實作類必須能完全替換介面/基底類別而不引發異常行為。
- **I (Interface Segregation)**: 
  - 客戶端不應依賴它用不到的介面，將龐大介面拆解為小巧具體的介面。
- **D (Dependency Inversion)**: 
  - 高階模組不應依賴低階模組，兩者皆應依賴抽象。
- **組合優於繼承 (Composition over Inheritance)**：優先使用物件組合或策略模式。

---

## 5. 錯誤處理 (Error Handling)

### 5.1 絕不吞掉 Exception (Never Swallow Exceptions)
- ❌ 嚴格禁止空的 `except Exception: pass` 或默默回傳假資料以偽裝成功。
- ✅ 捕捉例外後，必須記錄日誌、包裝為領域例外並向上拋出，或透過明確的 Failure 物件回傳。

### 5.2 提早失敗 (Fail Fast)
- 在進入核心業務前驗證輸入參數（例如輸入形狀、檔案路徑存在性），一旦無效立即拋出明確的自定義例外（如 `ValueError`, `FileNotFoundError`）。

---

## 6. 註解與程式碼整潔 (Comments & Formatting)

### 6.1 不要替壞代碼寫註解，直接重寫 (Don't Comment Bad Code, Rewrite It)
- 好的程式碼自帶解釋（Self-Documenting Code）。
- 註解僅用於解釋「**為什麼 (Why)** 這樣做」（業務原因、演算法背景、論文公式參考），而非解釋「**在做什麼 (What)**」。

### 6.2 嚴禁保留被註解的廢棄代碼 (Delete Dead Code)
- 任何被 `#` 或 `//` 註解掉的舊邏輯、未使用的 import、未使用的區域變數，**一律直接刪除**。Git 歷史記錄會完整保留過往版本。

---

## 7. 簡潔至上 (KISS & DRY)

- **DRY (Don't Repeat Yourself)**: 抽取重複邏輯為可共用的純函式或模組。
- **KISS (Keep It Simple, Stupid)**: 拒絕過度設計（Over-engineering），以最清晰直觀的方式解決問題。
- **童子軍法則 (Boy Scout Rule)**: 離開時讓營地比你抵達時更乾淨（每次修改檔案時，順手整理碰到的壞味道）。
