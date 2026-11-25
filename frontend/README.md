# Frontend

From `frontend`, install the locked dependencies and start the Vite development
server:

```powershell
npm ci
npm run dev
```

The backend API defaults to `http://localhost:8000/api/v1`. Override it with
`VITE_API_BASE_URL`; set `VITE_WS_BASE_URL` when the WebSocket server uses a
different host. The Vite server runs on port 5173.

Create a production bundle with `npm run build`.
