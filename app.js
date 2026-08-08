const budgetConfigurations = {
  1: {
    tokens: 8,
    stages: "1 / 4",
    label: "Gist",
    active: 4,
    description: "Gist stage는 장면과 주행동을 유지한다. 가장 작은 budget에서도 독립적으로 유효해야 한다.",
  },
  2: {
    tokens: 24,
    stages: "2 / 4",
    label: "Gist + events",
    active: 8,
    description: "Predictive residual이 큰 event만 한 단계 더 처리해 순서와 상태 변화를 보강한다.",
  },
  3: {
    tokens: 64,
    stages: "3 / 4",
    label: "Events + objects",
    active: 12,
    description: "선택된 event 안에서 object identity와 interaction을 위한 spatial token을 추가한다.",
  },
  4: {
    tokens: 128,
    stages: "4 / 4",
    label: "Full detail",
    active: 16,
    description: "가장 어려운 region만 deep sparse block까지 진행해 fine motion과 appearance detail을 복원한다.",
  },
};

const budgetRange = document.querySelector("#budget-range");
const budgetVisual = document.querySelector("#budget-visual");
const tokenCount = document.querySelector("#token-count");
const stageCount = document.querySelector("#stage-count");
const representationLabel = document.querySelector("#representation-label");
const budgetDescription = document.querySelector("#budget-description");

function buildBudgetTokens() {
  const fragment = document.createDocumentFragment();

  for (let index = 0; index < 16; index += 1) {
    const token = document.createElement("span");
    token.className = "budget-token";
    token.style.height = `${28 + (index % 4) * 9}px`;
    fragment.append(token);
  }

  budgetVisual.append(fragment);
}

function updateBudget(value) {
  const configuration = budgetConfigurations[value];
  const tokens = [...budgetVisual.children];

  tokenCount.textContent = configuration.tokens;
  stageCount.textContent = configuration.stages;
  representationLabel.textContent = configuration.label;
  budgetDescription.textContent = configuration.description;

  tokens.forEach((token, index) => {
    token.className = "budget-token";
    if (index < configuration.active) token.classList.add("is-active");
    if (index >= 4 && index < configuration.active) token.classList.add("is-refinement");
    if (index >= 12 && index < configuration.active) token.classList.add("is-detail");
  });
}

buildBudgetTokens();
updateBudget(budgetRange.value);
budgetRange.addEventListener("input", (event) => updateBudget(event.target.value));

const tabs = [...document.querySelectorAll('[role="tab"]')];

function activateTab(selectedTab) {
  tabs.forEach((tab) => {
    const isSelected = tab === selectedTab;
    const panel = document.querySelector(`#topic-panel-${tab.dataset.topic}`);
    tab.setAttribute("aria-selected", String(isSelected));
    panel.hidden = !isSelected;
    panel.classList.toggle("is-active", isSelected);
  });
}

tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => activateTab(tab));
  tab.addEventListener("keydown", (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;

    event.preventDefault();
    const direction = event.key === 'ArrowRight' ? 1 : -1;
    const nextIndex = (index + direction + tabs.length) % tabs.length;
    tabs[nextIndex].focus();
    activateTab(tabs[nextIndex]);
  });
});
