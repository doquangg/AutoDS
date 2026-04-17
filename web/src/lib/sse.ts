import type { PipelineEvent } from "./events";

/**
 * Connect to /sessions/{id}/events. Uses native EventSource so the browser
 * automatically handles reconnection and Last-Event-ID.
 *
 * Emits console diagnostics for every event, error, and lifecycle change
 * so a flaky stream is obvious from devtools rather than "UI just sits
 * there with nothing happening."
 */
export function subscribe(
  sessionId: string,
  onEvent: (ev: PipelineEvent) => void,
  onError: (e: Event) => void = () => {},
): () => void {
  const url = `/sessions/${sessionId}/events`;
  const es = new EventSource(url);
  const tag = `[sse ${sessionId.slice(0, 8)}]`;

  es.addEventListener("open", () => {
    console.log(`${tag} open readyState=${es.readyState}`);
  });

  es.onmessage = (e) => {
    try {
      const parsed = JSON.parse(e.data) as PipelineEvent;
      console.log(
        `${tag} recv id=${e.lastEventId} seq=${parsed.seq} type=${parsed.type}`,
      );
      onEvent(parsed);
    } catch (err) {
      console.error(`${tag} bad payload`, err, e.data);
    }
  };

  es.onerror = (e) => {
    // readyState: 0 = CONNECTING (auto-retry in progress), 2 = CLOSED
    console.warn(`${tag} error readyState=${es.readyState}`, e);
    onError(e);
  };

  return () => {
    console.log(`${tag} close (client)`);
    es.close();
  };
}
