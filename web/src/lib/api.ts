export async function createSession(
  file: File,
  query: string,
): Promise<{
  session_id: string;
  rows: number;
  cols: number;
  filename: string;
}> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("user_query", query);
  const r = await fetch("/sessions", { method: "POST", body: fd });
  if (!r.ok) throw new Error(`POST /sessions ${r.status}: ${await r.text()}`);
  return r.json();
}

export async function resumeSession(
  sid: string,
  target_column: string,
): Promise<void> {
  const r = await fetch(`/sessions/${sid}/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target_column }),
  });
  if (!r.ok) throw new Error(`POST /resume ${r.status}: ${await r.text()}`);
}

export function askQuestion(
  sid: string,
  question: string,
): Promise<Response> {
  return fetch(`/sessions/${sid}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}
