import type { PipelineEvent } from "./events";

/**
 * Connect to /sessions/{id}/events. Uses native EventSource so the browser
 * automatically handles reconnection and Last-Event-ID.
 */
export function subscribe(
  sessionId: string,
  onEvent: (ev: PipelineEvent) => void,
  onError: (e: Event) => void = () => {},
): () => void {
  const es = new EventSource(`/sessions/${sessionId}/events`);
  es.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data) as PipelineEvent);
    } catch (err) {
      console.error("Bad SSE payload", err, e.data);
    }
  };
  es.onerror = onError;
  return () => es.close();
}
