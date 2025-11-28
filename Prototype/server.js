// server.js (ES module)
import express from "express";
import multer from "multer";
import cors from "cors";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json({ limit: "100mb" }));

// uploads dir
const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

// Multer storage (v2)
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const timestamp = Date.now();
    const safeName = file.originalname.replace(/[^a-zA-Z0-9.\-_]/g, "_");
    cb(null, `${timestamp}-${Math.floor(Math.random()*1e6)}-${safeName}`);
  },
});
const upload = multer({ storage });

// serve public
const publicDir = path.join(__dirname, "public");
if (fs.existsSync(publicDir)) app.use(express.static(publicDir));

// POST /upload -> returns uploadedFiles array
app.post("/upload", upload.array("files"), (req, res) => {
  try {
    if (!req.files || req.files.length === 0) return res.json({ ok: false, error: "No files uploaded." });
    const uploadedFiles = req.files.map(f => path.join("uploads", path.basename(f.path)));
    return res.json({ ok: true, uploadedFiles });
  } catch (err) {
    console.error("Upload error:", err);
    return res.status(500).json({ ok: false, error: String(err) });
  }
});

// helper: run python script with JSON stdin and parse JSON stdout
function runPythonScript(scriptName, inputObj) {
  return new Promise((resolve, reject) => {
    const pythonExe = process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");
    const scriptPath = path.join(__dirname, scriptName);
    if (!fs.existsSync(scriptPath)) return reject(new Error(`${scriptName} not found at ${scriptPath}`));

    const py = spawn(pythonExe, [scriptPath], { stdio: ["pipe", "pipe", "pipe"], env: process.env });

    let stdout = "";
    let stderr = "";

    py.stdout.on("data", (c) => { stdout += c.toString(); });
    py.stderr.on("data", (c) => { stderr += c.toString(); });

    py.on("error", (err) => reject(new Error("Failed to start python: " + err.message)));

    py.on("close", (code) => {
      if (stderr && stderr.trim().length > 0) console.error("Python stderr:", stderr);
      if (!stdout || stdout.trim().length === 0) return reject(new Error("Python returned empty output"));
      try {
        const parsed = JSON.parse(stdout);
        resolve(parsed);
      } catch (err) {
        // try trimming and parse
        try {
          const txt = stdout.trim();
          const parsed = JSON.parse(txt);
          resolve(parsed);
        } catch (err2) {
          reject(new Error("Failed to parse python JSON output: " + err2.message + "\nstdout:\n" + stdout + "\nstderr:\n" + stderr));
        }
      }
    });

    try {
      py.stdin.write(JSON.stringify(inputObj));
    } catch (err) {
      // ignore
    } finally {
      py.stdin.end();
    }
  });
}

// POST /generate - call your existing generator (generator.py) if present
// Expect generator.py to accept JSON on stdin and return JSON (see earlier server in conversation)
app.post("/generate", async (req, res) => {
  const body = req.body || {};
  if (!body.numQuestions || !body.qType) return res.status(400).json({ ok: false, error: "Missing numQuestions or qType" });

  // convert docPaths to absolute
  const docPaths = Array.isArray(body.docPaths) ? body.docPaths.map(p => path.join(__dirname, p)) : [];

  const payload = {
    action: "generate",
    docPaths,
    numQuestions: Number(body.numQuestions),
    qType: body.qType,
    useCustomProportions: !!body.useCustomProportions,
    customProportions: body.customProportions || null
  };

  try {
    const pyRes = await runPythonScript("generator.py", payload);
    return res.json(pyRes);
  } catch (err) {
    console.error("GENERATE ERROR:", err);
    return res.status(500).json({ ok: false, error: err.message || String(err) });
  }
});

// POST /evaluate -> call report.py with { test, submission } and return { ok:true, report: "..." }
app.post("/evaluate", async (req, res) => {
  const body = req.body || {};
  if (!body.test || !body.submission) return res.status(400).json({ ok: false, error: "Missing test or submission in body" });

  try {
    const pyRes = await runPythonScript("report.py", { action: "report", test: body.test, submission: body.submission });
    // Expect pyRes to be { ok:true, report: "..." }
    if (!pyRes || !pyRes.ok) return res.status(500).json({ ok: false, error: (pyRes && pyRes.error) ? pyRes.error : 'report generation failed' });
    return res.json(pyRes);
  } catch (err) {
    console.error("GENERATE ERROR:", err);
    return res.status(500).json({ ok: false, error: err.message || String(err) });
  }
});

// POST /submit-answers -> save answers
app.post("/submit-answers", (req, res) => {
  const { sessionId, answers } = req.body || {};
  if (!sessionId || !answers) return res.status(400).json({ ok: false, error: "Missing sessionId or answers" });

  const outDir = path.join(__dirname, "results");
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `${sessionId}-${Date.now()}.json`);
  try {
    fs.writeFileSync(outFile, JSON.stringify({ sessionId, answers, receivedAt: new Date().toISOString() }, null, 2), "utf-8");
    return res.json({ ok: true, file: outFile });
  } catch (err) {
    console.error("Save answers error:", err);
    return res.status(500).json({ ok: false, error: String(err) });
  }
});

// static serve & fallback
app.use("/uploads", express.static(uploadDir));
if (fs.existsSync(publicDir)) app.use(express.static(publicDir));
app.get("*", (req, res) => {
  const indexFile = path.join(publicDir, "index.html");
  if (fs.existsSync(indexFile)) return res.sendFile(indexFile);
  return res.send("No frontend found. Put index.html into public/");
});

const PORT = process.env.PORT ? Number(process.env.PORT) : 5000;
app.listen(PORT, () => console.log(`Server listening on http://localhost:${PORT}`));
