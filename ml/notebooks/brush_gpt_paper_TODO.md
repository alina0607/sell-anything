# Brush GPT 論文:待辦清單(TODO)

> 對象:`brush_gpt_paper.ipynb`。
> 策略:**現在先給教授看 notebook 版**,但寫法盡量照論文標準,之後轉 Overleaf 時只需要「搬」,不必「重寫」。
> 每一項標了它屬於哪個階段:
> **[現在]** = 給教授看之前就該處理;**[轉論文]** = 轉 Overleaf 時再處理(現在先別動,免得教授版讀起來缺東西)。

---

## ✅ 已完成(2026-10-18)

數字與公式全部對過程式碼(`catbrush.py`、`brushtokens.py`、`brushgpt.py`、cat-face-landmarks 的
`mae.py`、`pose.py`、`label.py`、`finetune.py`、`landmarks.py`),修正如下:

| 位置 | 問題 | 怎麼改 |
|---|---|---|
| §1.2、§4.2、§4.3.3、Figure 8c/8d 圖說 | 3,046 和 3,064 兩個數字都出現,但沒說差在哪,讀起來像打錯字 | 兩個都對:**3,046 = 沒有臉部點**(trial、A–D、F、H);**3,064 = 有臉部點**(E、G、I、J)。每一處都標清楚是哪一種 |
| §4.2 版面公式 | 角度寫成 $\theta_i$,但 §4.1.2 寫 $\phi_i$,§5.1 的 $p_\theta$ 又把 θ 當模型參數 | 統一成 $\phi_i$ |
| §4.1.3 疊色公式 | `clip(c_i S_i)` 少了 `0, 1`(§4.1.2 有寫);$\prod_{j>i}$ 沒有上限 | 補成 `clip(c_i S_i, 0, 1)` 和 $\prod_{j>i}^{n_k}$ |
| §3.2 MAE 公式 | 沒說 var 是哪一種 | 註明是不偏樣本變異數(除以 767),跟程式的 `torch.var` 一致 |
| §3.4 | 規則排除後剩 6,830 隻,但 Table 3 的 CAT 是 6,848 隻,差 18 | 補一句:6,983 隻畫完的貓裡有 18 隻**沒有眼睛分數**,規則檢查不到,所以留在資料裡 → 6,983 − 135 = 6,848 |

核對過、**沒有錯**的地方:J 那一格「Full model scores 3.933」是 I 在 AFHQ val(3.9202,292 隻)和
CAT val(3.9407,521 隻)上按貓數加權的結果(3.9334),跟 D 的 3.9331 只是巧合。

---

## ✅ 已完成(2026-10-18,第二輪)

- **Table 2d 的標籤**:`kept` → `kept (of the paintings with eye measures)`。
  ⚠️ 只改了標籤文字(程式碼和存下來的輸出都改了,數字沒動),**還沒在 Mac 上重跑**。
- **`catbrush.py` 的 docstring**:三處「along the fur」改成「along the photo's edges」,跟論文 §4.1.4 一致(程式行為沒變)。
- **正文的開發日誌痕跡**:日期、「development log」、「project owner's agreement」、「hand-off note 7,531」都清掉了。
  被拿掉的出處**沒有丟**,記在下面「Appendix E 要列的出處」,寫附錄時照抄。
- **18 隻沒有眼睛分數的 CAT 畫作**:新增 `ml/tools/backfill_cat_eye_checks.py` 補算(見下面「要在 Mac 上跑的」)。
  從程式碼推斷的原因:`paint_cat_dataset.py` 從 10-12 03:58 開始畫,眼睛檢查是 10-12 才決定加的,
  **最早畫的那批紀錄裡沒有 `eye_conf`**;而 `brushgpt._painted()` 只有在 `"eye_conf" in e` 時才套規則,所以這批直接通過。
  ⚠️ 這是推斷,沒有資料無法證實;跑腳本時它會把那幾隻列出來,可以對照它們的畫圖時間。

## ✅ 已在 Mac 上完成(2026-10-17)

- `backfill_cat_eye_checks.py` 列出**剛好 18 隻**,`--write` 補算後改寫 `log.jsonl`(舊檔在 `log.jsonl.bak`)。
  只有這 18 行改變,而且只多了四個眼睛欄位,其餘內容不變。
- **推斷已證實**:這 18 張正是 log 最前面的畫作(第 0–18 行),第一行有 `eye_conf` 的是第 20 行,之後沒有任何畫作缺少。
- **18 隻全部通過規則**:排除數仍是 135(1.9%),保留 6,983 − 135 = **6,848**,正好等於 Table 3 的 CAT 貓數。
  Table 2c 的百分位、Figure 6 的數字(比值中位數 0.995、eye_err 0.038 / 0.104)與 Figure 7 的六隻貓都沒有變。
- 因此 **J 的資料(補算前 tokenise)與規則現在保留的貓完全相同**,§3.4 已照此改寫;Table 2d 的標籤改回 `kept`。

## 📎 Appendix E 要列的出處(從正文拿掉的日期與紀錄)

| 正文位置 | 原本寫的出處 |
|---|---|
| §3.2「與事前預期相反的結果」 | development log 2026-10-10 §1 |
| §3.3 重現性說明(Table 2 的 AUROC) | development log;2026-10-10 用未 commit 的臨時腳本算的 |
| Table 4a「Face box」 | decision 2026-10-10(臉部 vs 整張照片) |
| Table 4a「Brush shape」 | decision 2026-10-09(五種筆刷比較;程式在私人 Kaggle dataset `alinachuang/cat-brush-proto`) |
| §4.1.1 只畫臉 | 2026-10-10 決定 |
| §4.2 五 token 版面 | 2026-10-09 與 painter 一起決定 |
| §4.2 詞彙表 | 早期交接筆記寫 7,531(錯),程式為 7,499;development log 2026-10-11 記錄更正 |
| Table 6a、Table 6c 的 Trial 欄 | development log 2026-10-11 §2 |
| §4.3.3 field restriction | 2026-10-11 由專案擁有者決定採用(「這是架構改變還是補丁?」);正文改成指向 Appendix A |

## 🔴 優先處理
- [ ] **[轉論文] 文字的作者**:會議規定可以用 AI 潤稿,**不能由 AI 代寫段落和論點**。
      轉 Overleaf 時每一段用自己的話重寫,Claude 只負責審稿。
- [ ] **[轉論文] 「(made by Claude)」**:給教授看的版本保留(誠實揭露)。
      轉論文時從 §3.2、§4.1 標題拿掉,改寫進 **Acknowledgments / AI-use statement**,並查目標會議的 LLM policy。

## 🟠 範圍與實驗設計(審稿人一定會問)

- [ ] **RQ4(毛流)、RQ5(條件生成)還沒做完**:決定這篇只留 RQ1–RQ3,RQ4/RQ5 移到 Future Work、之後另寫一篇?
      (影響:摘要第 4 段、§1.3、§12、§13、Table 0 最後兩列。)
- [ ] **Judge 不夠獨立**:同一個 landmark model 同時負責標點 → 對齊(H)、定 eye-weight 區域(F)、過濾 CAT 資料,**又當評審**。
      方案:換一個不同架構或不同資料訓練的第二評審;至少在 Limitations 正面寫出來。
- [ ] **FID 樣本太少**(128/256 張,FID 通常要上萬張):小樣本時改以 KID 為主,或補更多樣本。
- [ ] **FID 的比較對象**:Table 0 是跟 *training* paintings 比;主結果應該改用 held-out **test**(摘要的 196 vs 49)。
- [ ] **每個版本只跑一個 seed**:「E、F 各自都在雜訊內,合起來才超過」需要多個 seed 才站得住。
- [ ] **沒有外部 baseline**(例如同資料的 pixel 模型、Sketch-RNN 式的連續輸出):投會議幾乎一定會被要求。
- [ ] **順序控制組**(§4.3.4 自己寫的):原順序 + field restriction 那一組還沒訓練。

## 🟡 參考文獻

- [ ] **[5] 和 [44] 是同一篇**(AttentionPainter)→ 合併,並更新正文的引用編號。
- [ ] [44]、[55]、[56]、[57] 用了**全形括號「(」**→ 改成半形(LaTeX 會出問題)。
- [ ] [11]「Wen et al.」缺作者名字縮寫。
- [ ] **每一筆都從 arXiv / DBLP 抓 BibTeX 重新核對**,尤其 2026 年的 arXiv 編號(2603.*、2605.*、2601.*)。
- [ ] §1–§4 沒有引用到 [22]–[31]、[39]、[44]–[52]、[55]–[59];等 §5 之後寫完再檢查一次,沒用到的刪掉。

## 🟡 notebook 痕跡(轉論文時移到附錄)

- [ ] **[轉論文] commit `d6e00d0`、檔名 `catbrush.py` / `modal_brushgpt.py` / `brushgpt.py`**:移到 Reproducibility 附錄。
- [ ] **[轉論文] 摘要裡的「Evidence rule … this notebook」**:改成一個附錄(Appendix E 本來就有)。
- [ ] **[轉論文] GitHub 連結 `alina0607/cat-face-landmarks`**:投雙盲會議時會洩漏身分,要匿名化;arXiv 可以保留。

## 🔵 轉 Overleaf 的結構(轉論文時)

- [ ] 模板:NeurIPS `preprint` 模式(單欄、顯示作者、不限頁數),之後要投別的會議只需要換 `main.tex`。
- [ ] 檔案:`main.tex` + `sections/*.tex`(一節一檔)+ `figures/*.pdf` + `refs.bib`。
- [ ] notebook 每張圖加一行 `plt.savefig("figures/xxx.pdf")` 輸出向量圖。
- [ ] 從「時間順序」改成「邏輯順序」:

| notebook | 論文去處 |
|---|---|
| Figure 2(一隻貓 0→608 筆) | 第一頁 teaser 圖 |
| Table 0 + Figure 1(12 列版本) | §實驗,改成 ablation 表 |
| §2 五張比較表 | 散文 + 只留 §2.6 那張 |
| §3.2 MAE / soft-argmax 公式 | 附錄 |
| §3.3 AUROC 重現性說明 | 附錄 |
| §4.1 painter、Table 4a | 主文只留 3 條式子,Table 4a 移到附錄 |
| §4.2 tokenisation + Table 4b | 主文保留 |
| §4.3.1 Trial 的故事 + Table 6c | 方法直接寫,Table 6c 移到 §實驗 |
| Figure 8c / 8d | 附錄 |

- [ ] 篇幅:會議主文約 9 頁(6,000–7,000 字),目前光是 §1–§4 就約 16,500 字 → 主要工作是挑選和搬到附錄。
- [ ] 寫作順序:§4 方法 → §3 資料 → §實驗 → §2 → §1 導論 → **摘要最後寫**。

## ℹ️ 投 arXiv 的實務

- [ ] 第一次在 cs 類別投稿通常需要 **endorsement** → 問 DigiPen 的教授。
- [ ] 分類:主類 **cs.CV**,cross-list cs.LG、cs.GR。
- [ ] 資料授權:AFHQ 是 CC BY-NC 4.0;CAT dataset 的授權要查;論文裡要寫明。
