const messageInput = document.getElementById('messageInput');
const predictButton = document.getElementById('predictButton');
const statusText = document.getElementById('statusText');
const resultCard = document.getElementById('resultCard');
const predictionBadge = document.getElementById('predictionBadge');
const hamScore = document.getElementById('hamScore');
const spamScore = document.getElementById('spamScore');
const hamBar = document.getElementById('hamBar');
const spamBar = document.getElementById('spamBar');

function toPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function updateResult(data) {
  const probabilities = data.probabilities || {};
  const ham = probabilities.ham || 0;
  const spam = probabilities.spam || 0;

  predictionBadge.textContent = data.prediction;
  predictionBadge.classList.toggle('spam', data.prediction === 'spam');
  hamScore.textContent = toPercent(ham);
  spamScore.textContent = toPercent(spam);
  hamBar.style.width = toPercent(ham);
  spamBar.style.width = toPercent(spam);
  resultCard.classList.remove('hidden');
}

async function analyzeMessage() {
  const message = messageInput.value.trim();
  if (!message) {
    statusText.textContent = 'Enter a message before running prediction.';
    resultCard.classList.add('hidden');
    return;
  }

  predictButton.disabled = true;
  statusText.textContent = 'Running model inference...';

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message })
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed.');
    }

    updateResult(data);
    statusText.textContent = 'Prediction complete.';
  } catch (error) {
    resultCard.classList.add('hidden');
    statusText.textContent = error.message;
  } finally {
    predictButton.disabled = false;
  }
}

predictButton.addEventListener('click', analyzeMessage);

messageInput.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
    analyzeMessage();
  }
});

document.querySelectorAll('[data-sample]').forEach((button) => {
  button.addEventListener('click', () => {
    messageInput.value = button.dataset.sample || '';
    messageInput.focus();
  });
});