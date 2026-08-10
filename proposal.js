const demos = document.querySelectorAll("[data-proposal-demo]");

const anytimeCopy = [
  ["Scene gist", "8 tokens", "장면과 주행동만 남기고 모든 clip이 shallow block에서 종료된다."],
  ["Event transitions", "24 tokens", "predictive residual이 큰 짧은 사건만 event block까지 진행한다."],
  ["Object states", "64 tokens", "선택된 사건 안에서 object identity와 state change를 보강한다."],
  ["Fine motion", "128 tokens", "가장 어려운 region만 마지막 sparse block까지 계산한다."],
];

const queryCopy = [
  ["Q1", "100%", "처음 질문은 global gist에서 Event B를 열어 증거와 계산을 cache한다."],
  ["Q2", "61%", "두 번째 질문은 Event B를 재사용하고 B2 leaf만 새로 연다."],
  ["Q3", "43%", "세 번째 질문은 B와 B2를 그대로 쓰고 Event C의 summary만 추가한다."],
  ["Q4", "34%", "관련된 후속 질문은 새 encoder 계산 없이 열린 evidence로 답한다."],
];

const latentCopy = [
  ["Observe", "Mₜ⁽⁰⁾", "새 visual evidence를 bounded world-state에 기록한다."],
  ["Predict", "Mₜ⁽¹⁾", "현재 belief로 다음 semantic state와 필요한 증거를 예측한다."],
  ["Compare", "Mₜ⁽²⁾", "예측과 새 관측의 불일치를 찾아 기존 belief를 검사한다."],
  ["Revise", "Mₜ⁽³⁾", "모순이 있으면 latent state를 수정하고 없으면 일찍 멈춘다."],
];

function setText(root, selector, value) {
  const element = root.querySelector(selector);
  if (element) element.textContent = value;
}

function updateAnytime(root, value) {
  const index = value - 1;
  const [label, tokens, description] = anytimeCopy[index];
  root.querySelectorAll(".demo-stage").forEach((stage, stageIndex) => {
    stage.classList.toggle("is-active", stageIndex <= index);
    stage.classList.toggle("is-current", stageIndex === index);
  });
  setText(root, "[data-demo-label]", label);
  setText(root, "[data-demo-value]", tokens);
  setText(root, "[data-demo-description]", description);
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

  update(demo, Number(input.value));
  input.addEventListener("input", (event) => update(demo, Number(event.target.value)));
});
