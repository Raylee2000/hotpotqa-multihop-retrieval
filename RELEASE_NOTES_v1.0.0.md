# Release v1.0.0 — HotpotQA Multi-hop Retrieval

> **狀態：預定發布（尚未建立 GitHub Release）。** 本文件是 `v1.0.0` Release 頁面的說明草稿；在 Release 實際建立並上傳附件前，模型尚未公開，文中沒有可用的下載網址。

## 版本對應

| 項目 | 版本 |
|:--|:--|
| GitHub Release／Git tag | `v1.0.0` |
| 研究程式碼與文件（README、`pyproject.toml`、ZIP 內 `NOTICE`／`PROVENANCE.json`） | Research Repository `v0.1.0` |
| 模型資產 | 檔名 `-v1`，`configs/models.json` 的 `asset_revision: license-metadata-1` |

`v1.0.0` 是首次公開發布的標籤；ZIP 內授權文件記載的 `v0.1.0` 指同一份研究程式碼，資產內容沒有因改用此標籤而變更。

## 研究範圍

- 任務：**HotpotQA Distractor 多跳文章檢索**——從每題提供的 2–10 篇候選文章中找出兩篇 Supporting Articles。
- 方法：General Encoder（第一跳）＋ SF Selector（預測支持句）＋ Conditional Second-Hop Encoder（CSH，第二跳）＋ 平行配對 → 27 項配對特徵（含 QType Bridge 機率）→ LambdaMART（`R5_full`）排序。
- 指標：Pair Recall@1（首位無序文章對完全等於兩篇 gold articles）。**不是** Answer EM/F1，也不含 LLM 答案生成。

### 歷史正式結果（PR@1，封存於 `results/formal/`、`results/reference/`）

| 方法 | V-report（8,000） | B/C（15,612） | Official Dev（7,405） |
|:--|--:|--:|--:|
| BM25 Local | 34.91% | 31.75% | 30.94% |
| Frozen Dense（未微調 MiniLM） | 33.41% | 41.72% | 32.98% |
| Hybrid Local（BM25 + Dense RRF） | 37.89% | 43.02% | 36.81% |
| General Top-2（Fine-tuned MiniLM） | 56.15% | 66.89% | 54.72% |
| **最終系統（R5_full）** | **90.96%** | **93.30%** | **89.32%**（6,614 / 7,405） |

這些是研究期間的正式評估紀錄；本發布版尚未在全新環境完成端到端重現（見「限制」）。

## 發布內容

### 1. Git Repository（tag `v1.0.0`）

程式碼（`run.py`、`scripts/`、`src/`、`tools/maintainer/`）、單元測試、設定與完整性 manifests（`configs/`、`data/manifests/`）、研究報告與正式結果（`results/`）、授權文件（`LICENSE`、`THIRD_PARTY_LICENSES.md`、`release_assets/legal/`）、`release_assets/SHA256SUMS` 與本文件。**不含**模型權重、ZIP、原始資料、cache 或執行輸出。

### 2. Release 附件（共 7 個檔案）

| 檔案 | Bytes | SHA256 | 內容 |
|:--|--:|:--|:--|
| `hotpotqa-general-encoder-v1.zip` | 83,680,074 | `45449f3d73796d6da5843fe5bd6db34586725abeb778a21f0b9b7aa740f29be0` | 第一跳 General Encoder（MiniLM 全參數微調，step 1869） |
| `hotpotqa-csh-v1.zip` | 83,675,610 | `1fd0d3fe79572e9d603d2957558465c836b65b36a4327be0da6bac0714f5da4e` | Conditional Second-Hop Encoder（step 1500） |
| `hotpotqa-sf-selector-fold0-v1.zip` | 83,619,179 | `c5801bd70ccb4e08575b7ed8ba312b8f06c5adcf9b6d98a8340c45657dbf9249` | SF Selector cross-encoder，fold 0（head 為 safetensors） |
| `hotpotqa-sf-selector-fold1-v1.zip` | 83,620,548 | `7de8f0196cfcdf9b7791a49e89faa61f5d485410c7e9e9d9c763fe9bdc5554d0` | SF Selector cross-encoder，fold 1 |
| `hotpotqa-qtype-frozen-lr-v1.zip` | 137,087 | `253e1d1df3457ed1481daf42bff0e364f309adb8994f8ad545c13661f43a95ff` | QType 凍結句向量 + Logistic Regression 參數（不含 MiniLM 權重） |
| `hotpotqa-r5-full-v1.zip` | 73,065 | `311bced4cda4de38c2012b24664700c15752f7f238844e5f722cbd38b06330cc` | R5_full LambdaMART：200 棵樹的數值陣列 + 27 特徵 schema（無 pickle） |
| `SHA256SUMS` | 572 | — | 上述六個 ZIP 的 SHA256（`sha256sum -c` 格式） |

每個 ZIP 的解壓檔案清單與逐檔 SHA256 記錄於 `configs/models.json`。只使用 safetensors、JSON、tokenizer 文字檔與 `allow_pickle=False` 可讀的 numpy 陣列。

## 安裝

需求：Python 3.11（`environment.yml`／`requirements-lock.txt`）。

```sh
conda env create -f environment.yml
conda activate RAG-HotpotQA
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
python run.py data --source hf                   # 固定 revision 的 HotpotQA，依 data lock 驗證
```

模型（二擇一）：

```sh
# A. 從 Release 頁面下載六個 ZIP 與 SHA256SUMS 至 release_assets/ 後
cd release_assets && sha256sum -c SHA256SUMS && cd ..
python run.py models --from-dir release_assets

# B. 由安裝器直接下載（OWNER/REPO 換成實際 Repository）
python run.py models --base-url https://github.com/OWNER/REPO/releases/download/v1.0.0
```

安裝器先比對 ZIP 大小與 SHA256，只解壓 manifest 所列的安全相對路徑並逐檔校驗；預設另從 Hugging Face 下載釘選的基礎模型 `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`（不在 ZIP 內）。之後：

```sh
python run.py check       # manifest、模型、基礎模型、資料 lock
python run.py smoke       # Official Dev 第一個 128 題 shard，並與封存結果比對
```

## 授權

- 原始碼：MIT（`LICENSE`）。
- 六個模型 ZIP：Apache-2.0。每個 ZIP 內附 `LICENSE`（Apache-2.0 全文）、`NOTICE`、`MODIFICATIONS.md`、`DATA_SOURCES.md`、`PROVENANCE.json`；四個 MiniLM 衍生模型（General、CSH、SF fold 0／1）另附未修改的上游 model card（`UPSTREAM_MODEL_CARD.md`）與 MiniLM 祖先模型的 MIT 授權全文及 Microsoft 版權聲明（`MINILM_MIT_LICENSE.txt`）。
- 訓練資料 HotpotQA：CC BY-SA 4.0，歸屬見各 ZIP 的 `DATA_SOURCES.md`。ZIP 不含 HotpotQA 題目、答案、文章內容、支持句紀錄、訓練 embedding 或候選特徵表。
- 詳見 `THIRD_PARTY_LICENSES.md`。

## 本次發布的驗證

- 單元測試 `python run.py test`、程式／設定完整性 `python run.py check --code-only`。
- 六個 ZIP：大小與 SHA256 對 `configs/models.json` 及 `SHA256SUMS`；ZIP 完整性；解壓項目集合與逐檔 SHA256；授權文件存在且與 `release_assets/legal/` 一致；`PROVENANCE.json` 記錄的原始檔 hash 對應 ZIP 內容；無 pickle 格式、無本機絕對路徑。
- 以維護者封裝工具 `tools/maintainer/package_assets.py` 重建六個 ZIP，結果與附件 **byte-identical**。
- 實際執行安裝器（`--from-dir` 與 `--base-url` 兩種來源，後者以本機檔案模擬下載），六個模型安裝並逐檔驗證通過。

## 限制

- 只評估題目提供候選文章的 **Distractor Retrieval**；不代表全 Wikipedia／open-domain 檢索能力，不生成答案。FullWiki、MDR 對照均未完成驗證。
- 上表為歷史正式結果。本發布版未在全新環境執行資料下載、模型推論、完整評估或完整訓練；端到端重現須依 README 實際執行。
- V-select 用於開發／選模；B/C 曾用於早期 QType 評估，不是完全未曝光的盲測。ID 分離不排除跨題文章重複、語義重疊或預訓練曝光。
- Official Dev 上 R5 相對單跳 baselines 的差距是整體系統差距；Official Dev 未跑 R0 或完整消融，不能歸因為純 reranker 增益。
- QType 權重是以相同條件重新擬合的替代品（原 `qtype_v2` 權重未保存）。
- `python run.py train` 重新訓練得到的不是發布權重，不保證與歷史分數逐位相同；`full`／`smoke` 端到端訓練在本次發布未執行。
- 依賴版本以原 Windows CPU 環境為準；其他 OS／架構（尤其 macOS 的 `+cpu` wheel）未驗證。
