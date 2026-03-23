import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type ChatMode = "document" | "gemini";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  meta?: {
    mode?: ChatMode;
    similarity?: number | null;
    sources?: SourceInfo[];
  };
};

type SourceInfo = {
  title: string;
  page_num: number;
  similarity: number;
  snippet: string;
};

type ChatResponse = {
  answer: string;
  mode: ChatMode;
  similarity?: number | null;
  sources?: SourceInfo[];
};

type DocumentItem = {
  title: string;
  source: string;
  type: string;
  chunk_count: number;
  created_at: string;
};

// const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'
const API_BASE =
  import.meta.env.VITE_API_BASE ||
  (window.location.hostname === "localhost" ? "http://localhost:8000" : "/api");
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
  const [isInputExpanded, setIsInputExpanded] = useState(false);
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

  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // Document Management & System State
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [isDocListOpen, setIsDocListOpen] = useState(false);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [toastMessage, setToastMessage] = useState<{
    text: string;
    type: "success" | "error";
  } | null>(null);
  const [backendStatus, setBackendStatus] = useState<
    "checking" | "online" | "offline"
  >("checking");

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

  const showToast = useCallback(
    (text: string, type: "success" | "error" = "success") => {
      setToastMessage({ text, type });
      setTimeout(() => setToastMessage(null), 5000);
    },
    [],
  );

  // Check Backend Health
  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        setBackendStatus("online");
      } else {
        setBackendStatus("offline");
      }
    } catch (e) {
      setBackendStatus("offline");
    }
  }, []);

  useEffect(() => {
    void checkHealth();
    const interval = setInterval(() => void checkHealth(), 30000); // Check every 30s
    return () => clearInterval(interval);
  }, [checkHealth]);

  const fetchDocuments = useCallback(async () => {
    setLoadingDocs(true);
    try {
      const res = await fetch(`${API_BASE}/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
        if (backendStatus === "offline") setBackendStatus("online");
      } else {
        setBackendStatus("offline");
      }
    } catch (e) {
      console.error("Failed to fetch documents", e);
      setBackendStatus("offline");
    } finally {
      setLoadingDocs(false);
    }
  }, [backendStatus]);

  useEffect(() => {
    void fetchDocuments();
  }, [fetchDocuments]);

  const deleteDocument = useCallback(
    async (title: string) => {
      if (
        !confirm(`'${title}' 문서를 바탕으로 저장된 내용을 삭제하시겠습니까?`)
      )
        return;
      try {
        const res = await fetch(
          `${API_BASE}/documents/${encodeURIComponent(title)}`,
          { method: "DELETE" },
        );
        if (res.ok) {
          showToast("해당 문서의 데이터가 삭제되었습니다.", "success");
          void fetchDocuments();
        } else {
          showToast("문서 삭제 거부됨 혹은 서버 에러.", "error");
        }
      } catch (e) {
        showToast("서버와 통신할 수 없습니다.", "error");
        setBackendStatus("offline");
      }
    },
    [fetchDocuments, showToast],
  );

  const canSend = useMemo(
    () => input.trim().length > 0 && !pending,
    [input, pending],
  );

  async function send() {
    const text = input.trim();
    if (!text || pending) return;

    if (backendStatus === "offline") {
      showToast(
        "현재 서버에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.",
        "error",
      );
      return;
    }

    setInput("");
    const userMsg: ChatMessage = { id: uid(), role: "user", text };
    setMessages((m) => [...m, userMsg]);

    setPending(true);
    try {
      console.log("[Chat] Sending message:", text);
      
      // 대화 히스토리 구성 (현재 메시지 제외)
      const history = messages.map((m) => ({
        role: m.role,
        content: m.text,
      }));
      
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history }),
      });
      console.log("[Chat] Response status:", res.status, res.ok);
      if (!res.ok) {
        const detail = await res.text();
        const errorMsg = detail || `HTTP ${res.status}`;
        console.error("[Chat] HTTP Error:", errorMsg);
        throw new Error(errorMsg);
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
      setBackendStatus("online");
      console.log("[Chat] Message completed successfully");
    } catch (e) {
      setBackendStatus("offline");
      const errorMsg = e instanceof Error ? e.message : String(e);
      console.error("[Chat] Network Error:", errorMsg, e);
      showToast(`답변 요청 중 오류: ${errorMsg}`, "error");
      const errorChatMsg: ChatMessage = {
        id: uid(),
        role: "assistant",
        text: `[오류]\n${errorMsg}`,
      };
      setMessages((m) => [...m, errorChatMsg]);
    } finally {
      setPending(false);
    }
  }

  const uploadPdf = useCallback(
    async (file: File) => {
      if (backendStatus === "offline") {
        showToast(
          "서버가 오프라인이므로 파일을 업로드할 수 없습니다.",
          "error",
        );
        return;
      }

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
          const err = await res
            .json()
            .catch(() => ({ detail: res.statusText }));
          throw new Error(err.detail || `업로드 실패 (${res.status})`);
        }
        setUploadError(null);
        showToast("문서 업로드 및 분석이 완료되었습니다!", "success");
        void fetchDocuments(); // Refresh document list
      } catch (e) {
        setUploadError(e instanceof Error ? e.message : "업로드 실패");
        showToast("문서 업로드 파일 전송 중 오류가 발생했습니다.", "error");
        setBackendStatus("offline");
      } finally {
        setUploading(false);
      }
    },
    [fetchDocuments, showToast, backendStatus],
  );

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

  const copyToClipboard = useCallback(
    async (text: string) => {
      try {
        await navigator.clipboard.writeText(text);
        showToast("복사되었습니다!", "success");
        console.log("[Copy] Copied to clipboard:", text.substring(0, 50));
      } catch (e) {
        showToast("복사 실패했습니다.", "error");
        console.error("[Copy] Failed:", e);
      }
    },
    [showToast],
  );

  // 어시스턴트 답변에서 프리픽스 제거 (출처 제외)
  const getAnswerTextOnly = useCallback((fullText: string): string => {
    // "[문서 참조 답변]\n" 또는 "[Gemini 추론 답변]\n" 제거
    return fullText.replace(/^\[(문서 참조|Gemini 추론) 답변\]\n/, "");
  }, []);

  return (
    <div className="page">
      {toastMessage && (
        <div className={`toast toast-${toastMessage.type}`}>
          {toastMessage.text}
        </div>
      )}

      <aside className={`sidebar ${isSidebarOpen ? "" : "closed"}`}>
        <div className="sidebarHeaderRow">
          <div className="sidebarTitle">
            <span className="sidebarIcon" aria-hidden>
              📄
            </span>
            문서 관리
          </div>
          <button
            type="button"
            className="closeSidebarBtn"
            onClick={() => setIsSidebarOpen(false)}
            aria-label="사이드바 닫기"
          >
            ×
          </button>
        </div>
        <p className="sidebarHint">PDF를 업로드하거나 삭제하세요</p>

        <div
          className={`dropZone ${dragOver ? "dropZoneActive" : ""}`}
          onClick={() => fileInputRef.current?.click()}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
        >
          <span className="dropZoneText">여기로 파일을 드래그하세요</span>
          <span className="dropZoneLimit">최대 {MAX_PDF_MB}MB 지원 (PDF)</span>
        </div>
        <button
          type="button"
          className="browseBtn"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? "문서 분석 중..." : "파일 찾아보기"}
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

        <div className="docListContainer">
          <button
            className="docListToggleBtn"
            onClick={() => setIsDocListOpen(!isDocListOpen)}
          >
            <span>📌 저장된 문서 보기 ({documents.length})</span>
            <span>{isDocListOpen ? "▲" : "▼"}</span>
          </button>

          {isDocListOpen && (
            <div className="docList">
              {loadingDocs ? (
                <span
                  className="sidebarHint"
                  style={{ textAlign: "center", display: "block" }}
                >
                  불러오는 중...
                </span>
              ) : documents.length === 0 ? (
                <span
                  className="sidebarHint"
                  style={{ textAlign: "center", display: "block" }}
                >
                  업로드된 문서가 없습니다.
                </span>
              ) : (
                <table className="docTable">
                  <thead>
                    <tr>
                      <th>제목</th>
                      <th>청크 수</th>
                      <th>생성일</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {documents.map((doc) => (
                      <tr key={doc.title}>
                        <td className="docItemTitle" title={doc.title}>
                          {doc.title}
                        </td>
                        <td className="docChunkVal">{doc.chunk_count}</td>
                        <td className="docDate">{doc.created_at}</td>
                        <td>
                          <button
                            className="docDeleteBtn"
                            onClick={() => void deleteDocument(doc.title)}
                            title="삭제하기"
                          >
                            🗑️
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="topbarLeft">
            {!isSidebarOpen && (
              <button
                type="button"
                className="toggleSidebarBtn"
                onClick={() => setIsSidebarOpen(true)}
                aria-label="사이드바 열기"
              >
                ≡
              </button>
            )}
            <div className="titles">
              <div
                className="titleRow"
                style={{ display: "flex", alignItems: "center", gap: "8px" }}
              >
                <h1 className="title">생산성 강화 RAG 챗봇</h1>
                <div className={`statusBadge ${backendStatus}`}>
                  <span className="statusDot" />
                  {backendStatus === "checking"
                    ? "연결 확인 중"
                    : backendStatus === "online"
                      ? "서버 정상"
                      : "연결 끊김"}
                </div>
              </div>
              <p className="subtitle">
                사내 규정, 매뉴얼 등 PDF를 업로드하고 질문해 보세요.
              </p>
            </div>
          </div>
          <button
            type="button"
            className="ghostBtn"
            onClick={clearHistory}
            disabled={pending || messages.length === 0}
          >
            대화 지우기
          </button>
        </header>

        <div className="chat" ref={listRef}>
          {messages.length === 0 ? (
            <div className="empty">
              <div className="emptyCard">
                <div className="emptyTitle">무엇이든 물어보세요!</div>
                <div className="emptyDesc">
                  왼쪽에 문서를 업로드하면 <b>[문서 참조 답변]</b>을,
                  <br />
                  업로드하지 않으면 <b>[AI 추론 답변]</b>을 제공합니다.
                </div>
                <div className="emptyHint">예: "사내 휴가 규정 요약해 줘."</div>
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
                          ? "🎯 문서 참조"
                          : "✨ AI 추론"}
                      </span>
                      {typeof m.meta.similarity === "number" ? (
                        <span className="sim">
                          정확도: {(m.meta.similarity * 100).toFixed(1)}%
                        </span>
                      ) : null}
                      {m.meta.sources && m.meta.sources.length > 0 ? (
                        <details className="sources">
                          <summary>💡 근거 출처 보기</summary>
                          <ul>
                            {m.meta.sources.map((s, idx) => (
                              <li key={idx} className="source-item">
                                <div className="source-header">
                                  <strong>{s.title}</strong>
                                  <span className="page-badge">
                                    p. {s.page_num}
                                  </span>
                                  <span className="similarity-badge">
                                    {(s.similarity * 100).toFixed(0)}% 유사
                                  </span>
                                </div>
                                <div className="source-snippet">
                                  "{s.snippet}"
                                </div>
                              </li>
                            ))}
                          </ul>
                        </details>
                      ) : null}
                    </div>
                  ) : null}
                </div>
                <button
                  type="button"
                  className="msgCopyBtn"
                  title="복사"
                  onClick={() => {
                    const textToCopy =
                      m.role === "assistant"
                        ? getAnswerTextOnly(m.text)
                        : m.text;
                    copyToClipboard(textToCopy);
                  }}
                >
                  📋
                </button>
              </div>
            ))
          )}
          {pending && (
            <div className="msgRow assistant">
              <div className="msgBubble">
                <div className="typing">
                  <span className="t" />
                  <span className="t" />
                  <span className="t" />
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="composerWrapper">
          <div className="composer">
            <div className="composerHeader">
              <button
                type="button"
                className={`expandBtn ${isInputExpanded ? "expanded" : ""}`}
                onClick={() => setIsInputExpanded(!isInputExpanded)}
                title={isInputExpanded ? "축소" : "확장"}
                aria-label={isInputExpanded ? "축소" : "확장"}
              >
                ⬍
              </button>
            </div>
            <div className="composerInputRow">
              <textarea
                className={`input ${isInputExpanded ? "expanded" : ""}`}
                rows={isInputExpanded ? 5 : 1}
                placeholder="메시지를 입력하세요 (Shift + Enter로 줄바꿈)"
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
          </div>
        </div>
      </main>
    </div>
  );
}
