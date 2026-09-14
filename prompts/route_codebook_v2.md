# Route annotation codebook, version 2

Frozen 2026-07-05, before any system comparison. This is the single codebook behind
**both** label sources in the paper: the silver route labels over the training pool
(assigned by Qwen3-32B, `scripts/silver_labels.py`) and the reference labels over the
EduLawQA-Route benchmark (assigned independently by three cross-family LLMs).

Each question receives exactly one label — **VECTOR / GRAPH / HYBRID** — plus the
sub-label **TEMPORAL** where it applies. The worked examples below were written fresh
for this codebook and are drawn from neither benchmark, so they cannot prime an
annotator toward benchmark answers.

Inter-annotator agreement measured on the benchmark was Cohen's kappa 0.27–0.53,
below the 0.65 target set before annotation. The GRAPH/HYBRID boundary is genuinely
blurred; the paper reports this as a finding. Treat these labels as a reference
signal, not ground truth.

The document is kept in its original Vietnamese, as used.

## Quy trình quyết định 2 CỔNG (trả lời tuần tự, dừng ở nhãn đầu tiên)

**G1 (cổng quan hệ):** *"Để trả lời ĐÚNG và ĐỦ, có bắt buộc đi qua ≥1 quan hệ liên-văn-bản tường minh (sửa đổi/bổ sung, thay thế/bãi bỏ, trạng thái hiệu lực, hướng dẫn/căn cứ, viện dẫn chéo) giữa ≥2 văn bản hoặc ≥2 điều khoản thuộc 2 văn bản khác nhau không?"*
→ **KHÔNG → VECTOR.** CÓ → sang G2.

**G2 (cổng nội dung):** *"Ngoài quan hệ đó, có cần trích nội dung nguyên văn điều khoản (định nghĩa, mức, điều kiện, thủ tục) để trả lời không?"*
→ **KHÔNG → GRAPH. CÓ → HYBRID.**

## Test thao tác nhanh từng nhãn

| Nhãn | Test 1 câu |
|---|---|
| VECTOR | "Nếu paste đúng 1-2 đoạn của MỘT điều luật vào prompt thì trả lời được?" → VECTOR |
| GRAPH | "Trả lời đúng bắt buộc đi ≥1 cạnh quan hệ giữa văn bản hoặc tra bảng hiệu lực?" → GRAPH |
| GRAPH-**T** (sub) | "Đổi mốc thời gian trong câu hỏi thì đáp án đổi?" (hiệu lực/sửa đổi/thay thế/bãi bỏ) → thêm sub=TEMPORAL |
| HYBRID | "Tách được thành ≥2 câu hỏi con thuộc 2 nhãn khác nhau?" → HYBRID |

## Quy tắc phụ (nguyên văn thiết kế)
1. Không chắc ≥80% → gán **HYBRID + cờ `ambiguous=true`** (câu ambiguous bị LOẠI khỏi routing-accuracy, GIỮ trong end-to-end).
2. Câu **should-refuse KHÔNG phải nhãn route** — refuse là hành vi tầng answer, đo riêng.
3. TEMPORAL là **sub-label của GRAPH/HYBRID**, không phải lớp thứ 4.

## Ví dụ VÀNG (soạn mới, không thuộc eval)

**VECTOR (5):**
1. "Mức trần học phí đại học công lập nhóm ngành y dược năm học 2025-2026 là bao nhiêu?" *(1 điều trong 1 nghị định)*
2. "Thời gian tập sự của giáo viên THCS hạng III là mấy tháng?"
3. "Điều kiện dự tuyển viên chức ngành giáo dục gồm những gì?"
4. "'Phổ cập giáo dục' được định nghĩa thế nào?"
5. "Thủ tục xin cấp bản sao bằng tốt nghiệp THPT gồm mấy bước?"

**GRAPH (5) — ≥3 mang sub TEMPORAL:**
1. "Thông tư 12/2011/TT-BGDĐT còn hiệu lực không?" *(T — tra bảng hiệu lực)*
2. "Nghị định nào thay thế Nghị định 46/2017/NĐ-CP?" *(T — cạnh SUPERSEDES)*
3. "Luật 34/2018/QH14 sửa đổi những luật nào?" *(T — cạnh AMENDS)*
4. "Điều 105 Luật Giáo dục 2019 được hướng dẫn bởi những nghị định nào?" *(cạnh hướng dẫn/căn cứ)*
5. "Hai thông tư về đánh giá học sinh tiểu học và THCS có quan hệ dẫn chiếu nhau không?" *(cấu trúc, không cần nội dung)*

**HYBRID (5):**
1. "Trường tôi đang áp dụng quy chế thi theo Thông tư cũ đã bị thay thế — quy định CHUYỂN TIẾP trong thông tư mới cho phép làm vậy đến khi nào?" *(quan hệ thay thế + nội dung điều chuyển tiếp)*
2. "So sánh mức phạt dạy thêm không phép giữa nghị định hiện hành và nghị định bị nó thay thế?" *(cạnh thay thế + nội dung 2 mức phạt)*
3. "Giáo viên hợp đồng trước 2021 có được xét tuyển đặc cách theo quy định hiện hành không, căn cứ điều khoản nào của cả văn bản cũ và mới?"
4. "Chuẩn hiệu trưởng hiện hành khác gì bản trước đó ở tiêu chí ngoại ngữ?" *(quan hệ phiên bản + nội dung 2 bản)*
5. "Học phí đã đóng theo mức của nghị định cũ thì có phải truy thu theo nghị định mới thay thế không?"

## Câu BẪY (dùng khi hiệu chỉnh annotator)

- *Bẫy tưởng-GRAPH-nhưng-VECTOR:* "Điều 5 và Điều 6 của cùng Luật X khác nhau thế nào?" → so sánh NỘI BỘ 1 văn bản = VECTOR (G1: không có quan hệ LIÊN văn bản).
- *Bẫy tưởng-VECTOR-nhưng-GRAPH-T:* "Quy chế tuyển sinh hiện hành là văn bản nào?" → "hiện hành" đòi tra hiệu lực = GRAPH-T dù nghe như lookup.
- *Bẫy HYBRID-giả:* "Nghị định X quy định gì và có hiệu lực từ ngày nào?" → ngày hiệu lực NẰM TRONG chính văn bản (điều cuối) = VECTOR, không cần quan hệ liên văn bản.

## Cam kết chống leak
Annotator nhìn câu eval CHỈ để gán nhãn; không ghi chép/nhắc lại câu eval khi soạn bất kỳ data train nào.
