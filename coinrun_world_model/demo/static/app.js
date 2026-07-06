const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const actionName = document.getElementById("actionName");
const modelState = document.getElementById("modelState");
const contextStrip = document.getElementById("context");
const temperature = document.getElementById("temperature");
const topk = document.getElementById("topk");
let keyToAction = {};

async function postJson(url, payload = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function drawPayload(payload) {
  const image = new Image();
  image.onload = () => {
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  };
  image.src = payload.frame;
  actionName.textContent = payload.action_name;
  modelState.textContent = payload.model_loaded ? "checkpoint loaded" : "mock mode";
  keyToAction = payload.key_to_action || keyToAction;
  contextStrip.innerHTML = "";
  for (const src of payload.context || []) {
    const img = document.createElement("img");
    img.src = src;
    contextStrip.appendChild(img);
  }
}

async function reset() {
  drawPayload(await postJson("/api/reset"));
}

async function step(action) {
  drawPayload(
    await postJson("/api/step", {
      action,
      temperature: Number(temperature.value),
      top_k: Number(topk.value),
    }),
  );
}

document.getElementById("reset").addEventListener("click", reset);
for (const button of document.querySelectorAll("button[data-action]")) {
  button.addEventListener("click", () => step(Number(button.dataset.action)));
}

window.addEventListener("keydown", (event) => {
  const action = keyToAction[event.code];
  if (action !== undefined) {
    event.preventDefault();
    step(action);
  }
});

reset();

