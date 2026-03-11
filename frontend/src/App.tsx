import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type ChatMode = "document" | "gemini";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  meta?: {
    mode?: ChatMode;
    similarity?: number | null;
    sources?: string[];
  };
};

type ChatResponse = {
  answer: string;
  mode: ChatMode;
  similarity?: number | null;
  sources?: string[];
};

// const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'
const API_BASE =
  window.location.hostname === "localhost" ? "http://localhost:8000" : "/api";
const STORAGE_KEY = "ragchat.messages.v1";
const MAX_PDF_MB = 200;

function uid() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function App() {
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw) as ChatMessage[];
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  });

  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    listRef.current?.scrollTo({
      top: listRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages.length, pending]);

  const canSend = useMemo(
    () => input.trim().length > 0 && !pending,
    [input, pending],
  );

  async function send() {
    const text = input.trim();
    if (!text || pending) return;

    setInput("");
    const userMsg: ChatMessage = { id: uid(), role: "user", text };
    setMessages((m) => [...m, userMsg]);

    setPending(true);
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as ChatResponse;
      const assistantMsg: ChatMessage = {
        id: uid(),
        role: "assistant",
        text: data.answer,
        meta: {
          mode: data.mode,
          similarity: data.similarity ?? null,
          sources: data.sources ?? [],
        },
      };
      setMessages((m) => [...m, assistantMsg]);
    } catch (e) {
      const assistantMsg: ChatMessage = {
        id: uid(),
        role: "assistant",
        text: `[오류]\n${e instanceof Error ? e.message : String(e)}`,
      };
      setMessages((m) => [...m, assistantMsg]);
    } finally {
      setPending(false);
    }
  }

  const uploadPdf = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("PDF 파일만 업로드 가능합니다.");
      return;
    }
    if (file.size > MAX_PDF_MB * 1024 * 1024) {
      setUploadError(`파일 크기는 ${MAX_PDF_MB}MB 이하여야 합니다.`);
      return;
    }
    setUploadError(null);
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_BASE}/ingest-pdf`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `업로드 실패 (${res.status})`);
      }
      setUploadError(null);
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : "업로드 실패");
    } finally {
      setUploading(false);
    }
  }, []);

  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) void uploadPdf(f);
      e.target.value = "";
    },
    [uploadPdf],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files?.[0];
      if (f) void uploadPdf(f);
    },
    [uploadPdf],
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  function clearHistory() {
    setMessages([]);
    localStorage.removeItem(STORAGE_KEY);
  }

  return (
    <div className="page">
      <aside className="sidebar">
        <div className="sidebarTitle">
          <span className="sidebarIcon" aria-hidden>
            📄
          </span>
          문서 업로드
        </div>
        <p className="sidebarHint">PDF 파일을 선택하세요</p>
        <div
          className={`dropZone ${dragOver ? "dropZoneActive" : ""}`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
        >
          <span className="dropZoneText">Drag and drop file here</span>
          <span className="dropZoneLimit">
            Limit {MAX_PDF_MB}MB per file - PDF
          </span>
        </div>
        <button
          type="button"
          className="browseBtn"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          Browse files
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          onChange={onFileChange}
          className="hiddenInput"
          aria-hidden
        />
        {uploadError && <p className="uploadError">{uploadError}</p>}
        {uploading && <p className="uploadStatus">업로드 중…</p>}
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="titles">
            <h1 className="title">생산성 강화 RAG 챗봇</h1>
            <p className="subtitle">
              PDF 문서를 업로드하고 관련된 질문을 해보세요! (관련 문서가 없을
              경우 기본 LLM 성능으로 답변합니다)
            </p>
          </div>
          <button
            type="button"
            className="ghostBtn"
            onClick={clearHistory}
            disabled={pending || messages.length === 0}
          >
            히스토리 삭제
          </button>
        </header>

        <div className="chat" ref={listRef}>
          {messages.length === 0 ? (
            <div className="empty">
              <div className="emptyCard">
                <div className="emptyTitle">질문을 입력해 보세요</div>
                <div className="emptyDesc">
                  유사도 0.7 이상 문서가 있으면 <b>[문서 참조 답변]</b>, 아니면{" "}
                  <b>[Gemini 추론 답변]</b>으로 시작합니다.
                </div>
                <div className="emptyHint">예: “사내 휴가 규정 요약해줘”</div>
              </div>
            </div>
          ) : (
            messages.map((m) => (
              <div key={m.id} className={`msgRow ${m.role}`}>
                <div className="msgBubble">
                  <pre className="msgText">{m.text}</pre>
                  {m.role === "assistant" && m.meta?.mode ? (
                    <div className="meta">
                      <span className={`badge ${m.meta.mode}`}>
                        {m.meta.mode === "document"
                          ? "문서 참조"
                          : "Gemini 추론"}
                      </span>
                      {typeof m.meta.similarity === "number" ? (
                        <span className="sim">
                          top sim: {m.meta.similarity.toFixed(3)}
                        </span>
                      ) : null}
                      {m.meta.sources && m.meta.sources.length > 0 ? (
                        <details className="sources">
                          <summary>참고 문서</summary>
                          <ul>
                            {m.meta.sources.map((s) => (
                              <li key={s}>{s}</li>
                            ))}
                          </ul>
                        </details>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              </div>
            ))
          )}
          {pending ? (
            <div className="msgRow assistant">
              <div className="msgBubble">
                <div className="typing">
                  <span className="t" />
                  <span className="t" />
                  <span className="t" />
                </div>
              </div>
            </div>
          ) : null}
        </div>

        <div className="composer">
          <textarea
            className="input"
            rows={2}
            placeholder="질문을 입력하세요..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
            disabled={pending}
          />
          <button
            type="button"
            className="sendBtn"
            onClick={() => void send()}
            disabled={!canSend}
            aria-label="전송"
          >
            <span className="sendArrow" aria-hidden>
              ↑
            </span>
          </button>
        </div>
      </main>
    </div>
  );
}
