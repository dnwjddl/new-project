const demos = document.querySelectorAll("[data-proposal-demo]");

const anytimeCopy = [
  ["Minimum persistence + innovation", "8 tokens", "첫 pair의 결합이 객체·관계·핵심 변화와 순서를 담는 semantic-temporal gist가 된다.", "Next Δ̂V = 0.31"],
  ["Transition refinement", "24 tokens", "첫 pair가 설명하지 못한 action phase와 long-horizon state transition을 다음 pair가 추가한다.", "Next Δ̂V = 0.14"],
  ["Rare-event coverage", "64 tokens", "짧고 드문 contact, direction reversal, object state change의 semantic residual을 보강한다.", "Next Δ̂V = 0.03 < threshold 0.06"],
  ["Residual semantic detail", "128 tokens", "남은 object relation과 fine motion만 추가하며 앞 token의 정보를 반복하지 않는다.", "Full path · no next stage"],
];

const queryCopy = [
  ["Q1", "100%", "처음 질문은 global gist에서 Event B를 열어 증거와 계산을 cache한다."],
  ["Q2", "61%", "두 번째 질문은 Event B를 재사용하고 B2 leaf만 새로 연다."],
  ["Q3", "43%", "세 번째 질문은 B와 B2를 그대로 쓰고 Event C의 summary만 추가한다."],
  ["Q4", "34%", "관련된 후속 질문은 새 encoder 계산 없이 열린 evidence로 답한다."],
];

const latentCopy = [
  ["Prior", "S⁻ₜ", "이전 posterior state에서 다음 bounded latent prior를 rollout한다."],
  ["Predict", "μₜ · σₜ", "여러 horizon의 future teacher latent distribution을 예측한다."],
  ["Innovation", "Iₜ", "실제 관측 residual을 uncertainty로 정규화해 semantic surprise를 계산한다."],
  ["Correct", "Sₜ", "innovation이 큰 slot만 반복 수정하고 충분히 수렴하면 일찍 멈춘다."],
];

function setText(root, selector, value) {
  const element = root.querySelector(selector);
  if (element) element.textContent = value;
}

function updateAnytime(root, value) {
  const mode = root.dataset.budgetMode || "fixed";
  const index = mode === "automatic" ? 2 : value - 1;
  const [label, tokens, description, marginal] = anytimeCopy[index];
  root.querySelectorAll(".demo-stage").forEach((stage, stageIndex) => {
    stage.classList.toggle("is-active", stageIndex <= index);
    stage.classList.toggle("is-current", stageIndex === index);
  });
  const control = root.querySelector("[data-budget-control]");
  const input = root.querySelector('input[type="range"]');
  control?.classList.toggle("is-disabled", mode === "automatic");
  if (input) input.disabled = mode === "automatic";
  setText(root, "[data-demo-label]", label);
  setText(root, "[data-demo-value]", tokens);
  setText(root, "[data-demo-description]", description);
  setText(root, "[data-demo-control]", mode === "automatic" ? "Progressive stop · tier 3" : `Fixed tier = ${index + 1}`);
  setText(root, "[data-demo-marginal]", mode === "automatic" ? marginal : "Gate logged; hard budget overrides it");
}

function updateQuery(root, value) {
  const index = value - 1;
  const [label, cost, description] = queryCopy[index];
  root.querySelectorAll(".query-node").forEach((node) => {
    const openedAt = Number(node.dataset.openAt || 1);
    node.classList.toggle("is-open", openedAt <= value);
    node.classList.toggle("is-new", openedAt === value);
  });
  setText(root, "[data-demo-label]", label);
  setText(root, "[data-demo-value]", cost);
  setText(root, "[data-demo-description]", description);
}

function updateLatent(root, value) {
  const index = value - 1;
  const [label, state, description] = latentCopy[index];
  root.querySelectorAll(".latent-state").forEach((node, nodeIndex) => {
    node.classList.toggle("is-active", nodeIndex <= index);
    node.classList.toggle("is-current", nodeIndex === index);
  });
  setText(root, "[data-demo-label]", label);
  setText(root, "[data-demo-value]", state);
  setText(root, "[data-demo-description]", description);
}

demos.forEach((demo) => {
  const input = demo.querySelector('input[type="range"]');
  const type = demo.dataset.proposalDemo;
  const update = type === "anytime" ? updateAnytime : type === "query" ? updateQuery : updateLatent;

  if (type === "anytime") {
    demo.dataset.budgetMode = "fixed";
    demo.querySelectorAll("[data-budget-mode]").forEach((button) => {
      button.addEventListener("click", () => {
        demo.dataset.budgetMode = button.dataset.budgetMode;
        demo.querySelectorAll("[data-budget-mode]").forEach((candidate) => {
          const active = candidate === button;
          candidate.classList.toggle("is-active", active);
          candidate.setAttribute("aria-pressed", String(active));
        });
        update(demo, Number(input.value));
      });
    });
  }

  update(demo, Number(input.value));
  input.addEventListener("input", (event) => update(demo, Number(event.target.value)));
});
