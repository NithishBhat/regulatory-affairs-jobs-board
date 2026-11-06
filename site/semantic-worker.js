// Background worker for the semantic step, so loading the model and embedding the résumé never freezes the page.
// A module worker: new Worker("semantic-worker.js", {type: "module"}).
// Messages in:  {run, text}
// Messages out: {type: "progress", e: {status, file, loaded, total}}
//               {type: "sims", run, ids: [job_id…], sims: Float32Array}   best-chunk cosine per job
//               {type: "error", run, message}
import "./match.js";                                 // sets self.Match
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const Match = self.Match;
let ready = null;
function load() {
  ready = ready || (async () => {
    const [meta, buf] = await Promise.all([
      fetch("vectors.json").then((r) => r.json()),
      fetch("vectors.bin").then((r) => r.arrayBuffer()),
    ]);
    const embed = await pipeline("feature-extraction", meta.model, {
      dtype: "q8",
      progress_callback: (e) => postMessage({ type: "progress", e: { status: e.status, file: e.file, loaded: e.loaded, total: e.total } }),
    });
    return { meta, rows: new Int16Array(buf), embed, ids: Object.keys(meta.index) };
  })();
  ready.catch(() => { ready = null; });             // allow a retry on the next résumé
  return ready;
}

onmessage = async ({ data }) => {
  try {
    const sem = await load();
    const vecs = (await sem.embed(Match.chunks(data.text), { pooling: "mean", normalize: true })).tolist();
    const sims = new Float32Array(sem.ids.length);
    sem.ids.forEach((id, i) => {
      const [first, count] = sem.meta.index[id];
      sims[i] = Match.maxSim(vecs, sem.rows, first, count, sem.meta.dim);
    });
    postMessage({ type: "sims", run: data.run, ids: sem.ids, sims }, [sims.buffer]);
  } catch (err) {
    postMessage({ type: "error", run: data.run, message: String((err && err.message) || err) });
  }
};
