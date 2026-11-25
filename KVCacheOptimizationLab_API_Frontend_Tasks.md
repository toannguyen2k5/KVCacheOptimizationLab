# KVCacheOptimizationLab — Backend API & Frontend Task Plan

> **Phạm vi:** triển khai lớp API ở backend và giao diện frontend để điều khiển, theo dõi và trình bày các thí nghiệm KV Cache.
>
> **Nguyên tắc bất biến:** model Qwen3-8B và phần XKV hiện tại được xem là các thành phần đã có, hoạt động như black-box. Không sửa, refactor, đổi tên, di chuyển hoặc tối ưu bất kỳ code hiện hữu nào trong `optimizers/xkv/`, các optimizer khác, model loader, inference engine, optimization engine, cache implementation hay metrics. Không thay đổi thuật toán hoặc luồng xử lý XKV.
>
> Chỉ tạo các file mới thuộc lớp API / frontend. Nếu cần kết nối với hàm hiện có, gọi qua interface công khai đang có. Nếu chưa có interface phù hợp, ghi nhận thành integration blocker; không tự ý sửa module hiện hữu để ép tương thích.

---

## 1. Phạm vi và quy ước

### Được phép thực hiện

- Tạo API routes, request/response schemas, API service/facade và WebSocket handler mới.
- Tạo frontend pages, components, hooks, services, types và cấu hình giao diện cần thiết.
- Viết test mới cho API và frontend.
- Viết tài liệu API và hướng dẫn chạy ứng dụng.

### Không được thực hiện

- Không sửa bất kỳ file nào trong `backend/optimizers/`, đặc biệt `backend/optimizers/xkv/`.
- Không sửa `backend/core/`, `backend/models/`, `backend/kv_cache/` hoặc các module thuật toán hiện có.
- Không thay đổi công thức XKV, CKA, SVD, reconstruction, cache representation hay quy trình decode.
- Không thay đổi weights, tokenizer, model config hoặc nội dung model.
- Không chạy formatter/refactor trên toàn dự án khiến các file ngoài phạm vi bị thay đổi.
- Không đổi tên hoặc di chuyển file hiện hữu.

### Quy tắc tích hợp

1. API là lớp điều phối, không phải nơi triển khai thuật toán.
2. Backend API chỉ gọi các chức năng hiện có thông qua public interface.
3. API không được tự tính lại hoặc mô phỏng kết quả XKV.
4. Nếu chưa xác định được chữ ký hàm hiện tại, cần kiểm tra code và ghi rõ interface cần dùng trước khi implement; không đoán tên hàm.
5. Frontend chỉ làm việc với API contract, không import trực tiếp Python/backend internals.
6. Các tác vụ chạy model phải chạy nền để không block request HTTP.

---

## 2. Kiến trúc mục tiêu

```text
Frontend (React + TypeScript)
        |
        | HTTP REST: cấu hình, tạo experiment, lấy kết quả
        | WebSocket: trạng thái và tiến trình chạy
        v
Backend API
  ├── REST Routes
  ├── Request / Response Schemas
  ├── API Services / Facade
  ├── Experiment Manager
  └── WebSocket Manager
        |
        | gọi public interface hiện có
        v
Existing Application Components (READ-ONLY)
  ├── Qwen3-8B model
  ├── Inference engine
  ├── Optimization engine
  ├── XKV optimizer
  └── KV Cache metrics / analyzer
```

`Existing Application Components` là vùng read-only trong phạm vi công việc này. API không được yêu cầu thay đổi nội bộ các thành phần đó.

### Cấu trúc file mới dự kiến

Đây là cấu trúc đề xuất; chỉ tạo file khi cần, không thay thế hoặc di chuyển file cũ.

```text
backend/
├── api/
│   ├── routes/
│   │   ├── health.py
│   │   ├── models.py
│   │   ├── experiments.py
│   │   ├── optimization.py
│   │   └── results.py
│   ├── schemas/
│   │   ├── common.py
│   │   ├── experiment.py
│   │   ├── optimization.py
│   │   └── result.py
│   ├── services/
│   │   ├── experiment_service.py
│   │   └── application_facade.py
│   └── websocket/
│       └── experiment_ws.py
│
frontend/
└── src/
    ├── components/
    │   ├── layout/
    │   ├── common/
    │   ├── charts/
    │   └── experiment/
    ├── pages/
    │   ├── Overview/
    │   ├── KVCacheAnalysis/
    │   ├── ExperimentSetup/
    │   ├── OptimizationLab/
    │   └── Evaluation/
    ├── services/
    │   ├── apiClient.ts
    │   ├── modelService.ts
    │   ├── experimentService.ts
    │   └── resultService.ts
    ├── hooks/
    │   ├── useExperiment.ts
    │   └── useExperimentSocket.ts
    └── types/
        ├── api.ts
        ├── experiment.ts
        └── metrics.ts
```

---

## 3. Backend API — Tasks

### BE-01 — API foundation

- [ ] Xác định framework API đang dùng và entry point hiện tại; không thay thế entry point nếu không cần thiết.
- [ ] Tạo router tổng hợp mới để đăng ký các route API.
- [ ] Thiết lập prefix thống nhất, đề xuất `/api/v1`.
- [ ] Thiết lập CORS cho địa chỉ frontend trong môi trường development.
- [ ] Chuẩn hóa response lỗi: `code`, `message`, `details`.
- [ ] Thêm endpoint kiểm tra trạng thái backend.
- [ ] Đảm bảo API khởi động không tự chạy inference hoặc load thêm model ngoài hành vi hiện có.

**Acceptance criteria**

- [ ] Backend khởi động được.
- [ ] `GET /api/v1/health` trả về trạng thái service.
- [ ] Swagger/OpenAPI hiển thị các endpoint đã đăng ký.
- [ ] Không có file thuật toán/model/XKV hiện hữu bị thay đổi.

### BE-02 — Model information API

- [ ] Tạo `GET /api/v1/models` để frontend biết model đang được cấu hình.
- [ ] Trả về metadata an toàn: model name, model status, device, dtype nếu có thể lấy từ interface hiện tại.
- [ ] Phân biệt rõ `loaded`, `not_loaded`, `loading`, `error`.
- [ ] Không trả về đường dẫn máy cá nhân, secret trong `.env` hoặc thông tin nhạy cảm.
- [ ] Không tự load/reload model từ endpoint này.

**Response đề xuất**

```json
{
  "items": [
    {
      "id": "qwen3-8b",
      "name": "Qwen3-8B",
      "status": "loaded",
      "device": "cuda:0",
      "dtype": "bfloat16"
    }
  ]
}
```

Metadata nào không lấy được từ trạng thái hiện có thì để `null` hoặc bỏ field; không suy đoán.

### BE-03 — Experiment configuration API

- [ ] Tạo schema cấu hình experiment.
- [ ] Tạo `GET /api/v1/experiments/config-options` để trả về các lựa chọn frontend được phép sử dụng.
- [ ] Tạo `POST /api/v1/experiments/validate` để validate prompt, generation parameters và cấu hình thí nghiệm.
- [ ] Validate kiểu dữ liệu, giới hạn token, giá trị rỗng và cấu hình không hợp lệ.
- [ ] Tách cấu hình experiment khỏi cấu hình model/runtime hiện tại.
- [ ] Không cho frontend gửi đường dẫn file tùy ý hoặc nội dung cấu hình có thể ghi đè `.env`.

**Các nhóm cấu hình dự kiến**

- Prompt / input text.
- Max new tokens.
- Temperature và sampling parameters (nếu inference interface hỗ trợ).
- Phương pháp chạy: baseline hoặc phương pháp tối ưu đã được backend công bố.
- Các tham số riêng của phương pháp chỉ được hiển thị khi interface hiện tại thực sự hỗ trợ.

### BE-04 — Create and run experiment API

- [ ] Tạo `POST /api/v1/experiments` để tạo experiment.
- [ ] Sinh `experiment_id` duy nhất.
- [ ] Lưu snapshot cấu hình đầu vào của experiment.
- [ ] Tạo trạng thái ban đầu: `queued`.
- [ ] Đưa tác vụ inference/optimization vào background worker hoặc executor phù hợp.
- [ ] API trả response ngay, không chờ toàn bộ quá trình generate hoàn thành.
- [ ] Tạo lớp `application_facade.py` mới để gọi public interface hiện tại.
- [ ] Không viết lại inference loop hoặc thuật toán XKV trong API service.
- [ ] Nếu chữ ký interface hiện tại không phù hợp, dừng ở facade và ghi integration blocker; không sửa module gốc.

**Experiment status chuẩn hóa**

`queued` → `running` → `completed`

Trạng thái kết thúc khác: `failed`, `cancelled`.

**Response đề xuất**

```json
{
  "experiment_id": "exp_...",
  "status": "queued",
  "created_at": "ISO-8601 timestamp"
}
```

### BE-05 — Experiment status and control API

- [ ] Tạo `GET /api/v1/experiments/{experiment_id}`.
- [ ] Trả về trạng thái, thời gian tạo, thời gian bắt đầu/kết thúc và cấu hình đã dùng.
- [ ] Tạo `GET /api/v1/experiments` hỗ trợ pagination và filter theo status.
- [ ] Xem xét `POST /api/v1/experiments/{experiment_id}/cancel` chỉ khi cơ chế chạy hiện tại hỗ trợ dừng an toàn.
- [ ] Không giả lập trạng thái cancel nếu backend không thể dừng tác vụ thực tế.
- [ ] Xử lý experiment ID không tồn tại bằng HTTP 404.

### BE-06 — Optimization API

- [ ] Tạo `GET /api/v1/optimization/methods` để công bố danh sách phương pháp thực sự được hỗ trợ.
- [ ] Mỗi method có `id`, `name`, `description`, `available` và danh sách tham số được phép cấu hình.
- [ ] Tạo `POST /api/v1/optimization/validate` để kiểm tra cấu hình phương pháp.
- [ ] Experiment service gọi optimization interface hiện có thông qua facade.
- [ ] Không implement lại XKV trong route/service.
- [ ] Không tự đặt tham số XKV nếu chưa đối chiếu với implementation hiện tại.
- [ ] Không quảng bá phương pháp là `available` nếu chưa tích hợp và chạy được.

### BE-07 — Result API

- [ ] Tạo `GET /api/v1/experiments/{experiment_id}/result`.
- [ ] Chuẩn hóa kết quả inference: generated text, token count và thời gian chạy nếu có.
- [ ] Chuẩn hóa metrics KV Cache theo dữ liệu thực tế được module hiện tại trả về.
- [ ] Phân biệt metric không có dữ liệu (`null`/omitted) với metric bằng 0.
- [ ] Không tự tính lại metric XKV trong API.
- [ ] Tạo `GET /api/v1/experiments/{experiment_id}/logs` nếu có log theo experiment.
- [ ] Kiểm tra experiment đã `completed` trước khi trả kết quả cuối cùng.

**Nhóm kết quả dự kiến**

- Inference: generated text, prompt tokens, generated tokens, latency.
- KV Cache memory: raw memory, compressed memory, memory saved, compression ratio.
- Reconstruction: reconstruction error/quality metrics nếu implementation có cung cấp.
- Analysis: CKA hoặc thông tin layer grouping nếu implementation có cung cấp.
- Runtime: peak VRAM/RAM, device và thời gian thực thi nếu có dữ liệu.

Tên metric và đơn vị phải thống nhất với output thực tế của backend; không tự tạo số liệu minh họa trong response production.

### BE-08 — WebSocket progress

- [ ] Tạo WebSocket endpoint `/ws/experiments/{experiment_id}`.
- [ ] Gửi event khi experiment đổi trạng thái.
- [ ] Gửi progress event nếu backend có tiến độ thực tế.
- [ ] Gửi log event có giới hạn kích thước và tần suất.
- [ ] Gửi result-ready event khi experiment hoàn tất.
- [ ] Gửi error event khi experiment thất bại.
- [ ] Hỗ trợ client reconnect; khi reconnect, client có thể gọi REST status API để đồng bộ.
- [ ] Không gửi tensor, KV Cache raw hoặc dữ liệu model lớn qua WebSocket.

**Event format đề xuất**

```json
{
  "event": "status",
  "experiment_id": "exp_...",
  "timestamp": "ISO-8601 timestamp",
  "data": {
    "status": "running",
    "message": "Experiment is running"
  }
}
```

Các event type đề xuất: `status`, `progress`, `log`, `result_ready`, `error`.

### BE-09 — Persistence and experiment history

- [ ] Xác định nơi lưu metadata experiment và kết quả đã chuẩn hóa.
- [ ] Ưu tiên lưu metadata, config, status và metrics; không lưu tensor KV Cache lớn nếu chưa có yêu cầu rõ ràng.
- [ ] Tạo repository/service mới cho persistence nếu cần.
- [ ] Tạo API xem lịch sử và chi tiết experiment.
- [ ] Xử lý trường hợp backend restart khi experiment đang chạy.
- [ ] Không ghi đè dữ liệu experiment cũ khi chạy experiment mới.

### BE-10 — API tests and documentation

- [ ] Test health endpoint.
- [ ] Test schema validation và lỗi request.
- [ ] Test experiment lifecycle bằng mock facade, không cần chạy model thật.
- [ ] Test trường hợp experiment ID không tồn tại.
- [ ] Test result API khi experiment đang chạy/chưa có kết quả.
- [ ] Test WebSocket connect, status event và error event.
- [ ] Viết tài liệu endpoint, request/response mẫu và mã lỗi.
- [ ] Có integration test riêng với model thật; không chạy mặc định trong unit test.

---

## 4. Frontend — Tasks

### FE-01 — Application foundation and layout

- [ ] Thiết lập routing cho 5 trang: Overview, KVCacheAnalysis, ExperimentSetup, OptimizationLab, Evaluation.
- [ ] Tạo layout chung: sidebar, header, main content.
- [ ] Tạo navigation và trạng thái active page.
- [ ] Tạo component loading, empty state, error state và confirmation dialog.
- [ ] Thiết lập theme và quy tắc spacing/typography thống nhất.
- [ ] Thiết kế responsive tối thiểu cho desktop và tablet.
- [ ] Không hardcode dữ liệu kết quả thí nghiệm vào component production.

### FE-02 — API client and shared types

- [ ] Tạo `apiClient.ts` làm đầu mối gọi backend.
- [ ] Đọc API base URL từ biến môi trường frontend.
- [ ] Chuẩn hóa xử lý HTTP error và timeout.
- [ ] Tạo TypeScript types khớp với backend schemas.
- [ ] Tạo service riêng cho model, experiment và result.
- [ ] Không gọi backend trực tiếp rải rác trong các page component.
- [ ] Tạo WebSocket hook có reconnect và cleanup khi component unmount.

### FE-03 — Overview page

- [ ] Hiển thị model hiện tại và trạng thái model.
- [ ] Hiển thị backend connection status.
- [ ] Hiển thị tổng số experiment và số experiment theo trạng thái nếu API cung cấp.
- [ ] Hiển thị các experiment gần đây.
- [ ] Tạo quick action dẫn đến Experiment Setup.
- [ ] Có empty state khi chưa có experiment nào.
- [ ] Không tự tạo số liệu dashboard khi API chưa có dữ liệu.

### FE-04 — KVCacheAnalysis page

- [ ] Tạo khu vực nhập/chọn experiment đã hoàn tất.
- [ ] Hiển thị thông tin prompt và generation metadata.
- [ ] Hiển thị cấu trúc KV Cache ở mức metadata: layer, head, sequence length, shape nếu backend cung cấp.
- [ ] Hiển thị memory trước/sau tối ưu nếu có dữ liệu.
- [ ] Hiển thị layer grouping/CKA nếu kết quả thực tế có cung cấp.
- [ ] Dùng chart/table có đơn vị và tooltip giải thích.
- [ ] Có trạng thái “metric chưa được backend cung cấp”.
- [ ] Không tính lại CKA, SVD hoặc reconstruction ở frontend.

### FE-05 — ExperimentSetup page

- [ ] Tạo form nhập prompt.
- [ ] Hiển thị model đang được cấu hình.
- [ ] Tạo form generation parameters theo config-options API.
- [ ] Cho phép chọn phương pháp từ methods API.
- [ ] Hiển thị tham số tương ứng với method đã chọn.
- [ ] Validate input phía frontend để cải thiện UX.
- [ ] Gọi validate API trước khi tạo experiment nếu cần.
- [ ] Khi submit, gọi create experiment API và nhận `experiment_id`.
- [ ] Chuyển người dùng sang màn hình theo dõi experiment sau khi tạo thành công.
- [ ] Chặn submit lặp khi request đang xử lý.

### FE-06 — OptimizationLab page

- [ ] Tạo màn hình chạy experiment và theo dõi tiến trình.
- [ ] Hiển thị experiment ID, status và thời gian chạy.
- [ ] Kết nối WebSocket theo experiment ID.
- [ ] Hiển thị progress chỉ khi backend gửi progress thực.
- [ ] Hiển thị log theo thời gian, có giới hạn số dòng hiển thị.
- [ ] Hiển thị generated text khi có kết quả.
- [ ] Hiển thị lỗi rõ ràng khi experiment thất bại.
- [ ] Có nút quay lại danh sách/lịch sử experiment.
- [ ] Chỉ hiển thị nút Cancel nếu backend hỗ trợ cancel thực tế.

### FE-07 — Evaluation page

- [ ] Cho phép chọn hai hoặc nhiều experiment đã hoàn tất để so sánh.
- [ ] Hiển thị bảng so sánh các metrics có cùng định nghĩa và đơn vị.
- [ ] Hiển thị memory saved, compression ratio, latency và reconstruction metrics nếu có.
- [ ] Cho phép chọn metric để hiển thị biểu đồ.
- [ ] Hiển thị cảnh báo khi hai experiment dùng prompt/config khác nhau.
- [ ] Không so sánh metric thiếu dữ liệu hoặc khác đơn vị như thể chúng tương đương.
- [ ] Cho phép export kết quả dạng JSON/CSV nếu API hỗ trợ hoặc frontend có thể export dữ liệu đã nhận.

### FE-08 — Reusable components

- [ ] `StatusBadge`: queued/running/completed/failed/cancelled.
- [ ] `MetricCard`: giá trị, đơn vị, mô tả và trạng thái thiếu dữ liệu.
- [ ] `ExperimentTable`: danh sách và trạng thái experiment.
- [ ] `PromptInput`: nhập prompt và đếm ký tự/token nếu có API hỗ trợ.
- [ ] `MetricChart`: chart dùng dữ liệu backend trả về.
- [ ] `LogViewer`: hiển thị log có giới hạn.
- [ ] `EmptyState`, `ErrorState`, `LoadingState`.
- [ ] `ConfirmDialog` cho thao tác có ảnh hưởng đến experiment.

### FE-09 — Frontend tests and quality

- [ ] Test API services bằng mocked HTTP responses.
- [ ] Test form validation và submit experiment.
- [ ] Test status rendering cho mọi trạng thái experiment.
- [ ] Test WebSocket hook: connect, message, reconnect và cleanup.
- [ ] Test empty/error/loading states.
- [ ] Chạy lint, typecheck và production build.
- [ ] Kiểm tra frontend không import hoặc phụ thuộc trực tiếp vào code Python.

---

## 5. API contract tổng hợp

| Method | Endpoint | Mục đích |
|---|---|---|
| GET | `/api/v1/health` | Kiểm tra backend |
| GET | `/api/v1/models` | Lấy trạng thái/metadata model |
| GET | `/api/v1/experiments` | Danh sách experiment |
| GET | `/api/v1/experiments/config-options` | Lấy lựa chọn cấu hình |
| POST | `/api/v1/experiments/validate` | Validate cấu hình |
| POST | `/api/v1/experiments` | Tạo và chạy experiment |
| GET | `/api/v1/experiments/{id}` | Trạng thái/chi tiết experiment |
| POST | `/api/v1/experiments/{id}/cancel` | Hủy nếu hỗ trợ an toàn |
| GET | `/api/v1/optimization/methods` | Danh sách phương pháp khả dụng |
| POST | `/api/v1/optimization/validate` | Validate tham số tối ưu |
| GET | `/api/v1/experiments/{id}/result` | Lấy kết quả |
| GET | `/api/v1/experiments/{id}/logs` | Lấy log |
| WS | `/ws/experiments/{id}` | Theo dõi realtime |

Đây là contract mục tiêu. Chỉ công bố endpoint khi đã implement; không để frontend phụ thuộc vào endpoint chưa tồn tại.

---

## 6. Thứ tự triển khai đề xuất

### Milestone 1 — API foundation

- [ ] BE-01 API foundation.
- [ ] BE-02 Model information API.
- [ ] FE-01 Application layout.
- [ ] FE-02 API client and shared types.

**Kết quả:** frontend chạy được, kết nối backend và hiển thị trạng thái hệ thống/model.

### Milestone 2 — Experiment lifecycle

- [ ] BE-03 Experiment configuration API.
- [ ] BE-04 Create and run experiment API.
- [ ] BE-05 Experiment status API.
- [ ] FE-03 Overview.
- [ ] FE-05 Experiment Setup.

**Kết quả:** người dùng có thể nhập prompt, cấu hình và khởi chạy experiment qua giao diện.

### Milestone 3 — Realtime Optimization Lab

- [ ] BE-06 Optimization API.
- [ ] BE-08 WebSocket progress.
- [ ] FE-06 OptimizationLab.
- [ ] FE-08 Reusable components.

**Kết quả:** người dùng theo dõi được trạng thái, log và kết quả sinh văn bản.

### Milestone 4 — Analysis and Evaluation

- [ ] BE-07 Result API.
- [ ] BE-09 Persistence/history.
- [ ] FE-04 KVCacheAnalysis.
- [ ] FE-07 Evaluation.

**Kết quả:** xem được metrics và so sánh các experiment đã chạy.

### Milestone 5 — Testing and delivery

- [ ] BE-10 API tests and documentation.
- [ ] FE-09 Frontend tests and quality.
- [ ] Kiểm thử end-to-end từ nhập prompt đến xem kết quả.
- [ ] Kiểm tra lại phạm vi thay đổi bằng Git diff.
- [ ] Xác nhận không có file thuộc vùng read-only bị sửa.

---

## 7. Definition of Done

Chỉ xem phần API + frontend hoàn thành khi:

- [ ] Frontend gọi backend qua REST API, không dùng dữ liệu giả làm kết quả thật.
- [ ] Người dùng có thể tạo một experiment từ giao diện.
- [ ] Backend chạy experiment thông qua interface hiện có, không viết lại thuật toán.
- [ ] Frontend theo dõi được trạng thái experiment.
- [ ] Kết quả và metrics hiển thị đúng dữ liệu backend trả về.
- [ ] Có xử lý loading, empty, error và reconnect.
- [ ] Có test cho các luồng API/frontend quan trọng.
- [ ] `git diff` xác nhận không có thay đổi trong `backend/optimizers/`, `backend/core/`, `backend/models/`, `backend/kv_cache/` và các file thuật toán hiện hữu.
- [ ] Không thay đổi model weights, XKV implementation hoặc cấu hình chạy model hiện tại.

## 8. Checklist bảo vệ phạm vi trước mỗi commit

```text
[ ] Tôi chỉ thêm/sửa file thuộc backend/api hoặc frontend.
[ ] Tôi không sửa backend/optimizers/xkv/.
[ ] Tôi không sửa backend/optimizers/ hoặc thuật toán optimizer khác.
[ ] Tôi không sửa backend/core/, backend/models/ hoặc backend/kv_cache/.
[ ] Tôi không thay đổi weights, model config hay runtime config hiện tại.
[ ] API chỉ gọi interface có sẵn; không copy/reimplement thuật toán.
[ ] Tôi đã kiểm tra git diff trước khi commit.
```

Lệnh kiểm tra nhanh trên PowerShell:

```powershell
git status --short
git diff -- backend/optimizers backend/core backend/models backend/kv_cache
```

Nếu lệnh thứ hai có output, cần xem xét và hoàn nguyên các thay đổi ngoài phạm vi trước khi commit.
