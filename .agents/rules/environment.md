# 虛擬環境與執行規範 (Virtual Environment Rule)

本專案所有的 Python 執行、腳本執行、測試與套件管理均必須在 Mamba 的 `hw1` 虛擬環境下運行。

## 核心規範

1. **強制使用 `hw1` 環境**：
   - 絕對禁止直接使用 base 環境或系統預設 Python。
   - 虛擬環境名稱：`hw1`
   - 虛擬環境路徑：`C:\Users\USER\AppData\Roaming\mamba\envs\hw1`

2. **終端與指令執行規範 (PowerShell)**：
   每次透過 `run_command` 或終端執行 Python 腳本、工具或測試時，必須採用以下方式之一：
   - **方式一（串聯啟動，推薦）**：
     ```powershell
     mamba activate hw1; python <script_path>
     ```
   - **方式二（指定環境執行）**：
     ```powershell
     mamba run -n hw1 python <script_path>
     ```
   - 若為套件安裝或環境檢查：
     ```powershell
     mamba activate hw1; pip install <package>
     # 或
     mamba run -n hw1 pip install <package>
     ```

3. **背景任務與子進程 (Subprocesses / Daemons)**：
   若需背景執行長時間任務或呼叫子進程，務必確保 Python 直譯器指向 `hw1` 環境（例如調用 `C:\Users\USER\AppData\Roaming\mamba\envs\hw1\python.exe` 或透過 `mamba run -n hw1`）。
