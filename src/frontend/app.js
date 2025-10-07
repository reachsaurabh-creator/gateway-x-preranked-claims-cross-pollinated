// Gateway X Frontend Application
const API_URL = 'http://localhost:3001';

// Elements
const queryInput = document.getElementById('query-input');
const submitBtn = document.getElementById('submit-btn');
const budgetSlider = document.getElementById('budget-slider');
const budgetValue = document.getElementById('budget-value');
const confidenceSlider = document.getElementById('confidence-slider');
const confidenceValue = document.getElementById('confidence-value');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const error = document.getElementById('error');
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const toggleDetailsBtn = document.getElementById('toggle-details-btn');
const detailedView = document.getElementById('detailed-view');
const roundsContainer = document.getElementById('rounds-container');

// Update slider values
budgetSlider.addEventListener('input', (e) => {
    budgetValue.textContent = e.target.value;
});

confidenceSlider.addEventListener('input', (e) => {
    confidenceValue.textContent = e.target.value;
});

// Toggle detailed view
let isDetailedViewVisible = false;
toggleDetailsBtn.addEventListener('click', () => {
    isDetailedViewVisible = !isDetailedViewVisible;
    
    if (isDetailedViewVisible) {
        detailedView.classList.remove('hidden');
        toggleDetailsBtn.querySelector('.btn-text').textContent = 'Hide Detailed View';
        toggleDetailsBtn.querySelector('.btn-icon').textContent = '👁️‍🗨️';
    } else {
        detailedView.classList.add('hidden');
        toggleDetailsBtn.querySelector('.btn-text').textContent = 'Show Detailed View';
        toggleDetailsBtn.querySelector('.btn-icon').textContent = '🔍';
    }
});

// Check server status
async function checkStatus() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            statusIndicator.classList.add('connected');
            statusText.textContent = 'Connected';
            submitBtn.disabled = false;
        } else {
            throw new Error('Server not responding');
        }
    } catch (err) {
        statusIndicator.classList.remove('connected');
        statusText.textContent = 'Disconnected';
        submitBtn.disabled = true;
    }
}

// Submit query
async function submitQuery() {
    const query = queryInput.value.trim();
    if (!query) {
        showError('Please enter a question');
        return;
    }
    
    // Hide previous results/errors
    results.classList.add('hidden');
    error.classList.add('hidden');
    loading.classList.remove('hidden');
    
    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                budget: parseInt(budgetSlider.value),
                confidence_threshold: parseFloat(confidenceSlider.value)
            })
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }
        
        const data = await response.json();
        showResults(data);
        
    } catch (err) {
        showError(err.message);
    } finally {
        loading.classList.add('hidden');
    }
}

// Show results
function showResults(data) {
    results.classList.remove('hidden');
    
    // Update result content
    document.querySelector('.best-claim').textContent = data.best_claim;
    document.querySelector('.confidence-value').textContent = 
        `${(data.confidence * 100).toFixed(1)}%`;
    document.querySelector('.rounds-value').textContent = data.rounds_used;
    
    // Show engines if available
    const enginesEl = document.querySelector('.engines-value');
    if (data.engines_used && data.engines_used.length > 0) {
        enginesEl.textContent = data.engines_used.join(', ');
    } else {
        enginesEl.textContent = 'Single engine';
    }
    
    // Render detailed view if data is available
    if (data.detailed_rounds && data.detailed_rounds.length > 0) {
        renderDetailedRounds(data.detailed_rounds);
    }
}

// Show error
function showError(message) {
    error.classList.remove('hidden');
    document.querySelector('.error-message').textContent = `Error: ${message}`;
}

// Event listeners
submitBtn.addEventListener('click', submitQuery);
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        submitQuery();
    }
});

// Render detailed rounds view
function renderDetailedRounds(rounds) {
    roundsContainer.innerHTML = '';
    
    rounds.forEach(round => {
        const roundDiv = document.createElement('div');
        roundDiv.className = 'round-container';
        
        const thresholdStatus = round.threshold_met ? 'threshold-met' : 'threshold-not-met';
        const statusText = round.threshold_met ? 'Threshold Met' : 'Threshold Not Met';
        
        roundDiv.innerHTML = `
            <div class="round-header">
                <div class="round-title">Round ${round.round_number}</div>
                <div class="round-status ${thresholdStatus}">${statusText}</div>
            </div>
            <div class="round-metrics">
                <div class="round-metric">
                    <span class="round-metric-label">Engines Used</span>
                    <span class="round-metric-value">${round.engines_used.length}</span>
                </div>
                <div class="round-metric">
                    <span class="round-metric-label">Confidence</span>
                    <span class="round-metric-value">${round.round_confidence ? (round.round_confidence * 100).toFixed(1) + '%' : 'N/A'}</span>
                </div>
                <div class="round-metric">
                    <span class="round-metric-label">Responses</span>
                    <span class="round-metric-value">${round.responses.length}</span>
                </div>
            </div>
            <div class="engine-responses" id="round-${round.round_number}-responses"></div>
        `;
        
        const responsesContainer = roundDiv.querySelector(`#round-${round.round_number}-responses`);
        
        // Sort responses by ranking (if available) or by score
        const sortedResponses = [...round.responses].sort((a, b) => {
            if (a.ranking !== null && b.ranking !== null) {
                return a.ranking - b.ranking;
            }
            if (a.score !== null && b.score !== null) {
                return b.score - a.score;
            }
            return 0;
        });
        
        sortedResponses.forEach((response, index) => {
            const responseDiv = document.createElement('div');
            responseDiv.className = `engine-response rank-${response.ranking || (index + 1)}`;
            
            const rankDisplay = response.ranking ? `#${response.ranking}` : `#${index + 1}`;
            const scoreDisplay = response.score ? response.score.toFixed(3) : 'N/A';
            const costDisplay = response.cost ? `$${response.cost.toFixed(4)}` : 'N/A';
            const tokensDisplay = response.tokens || 'N/A';
            const timeDisplay = response.response_time ? `${(response.response_time * 1000).toFixed(0)}ms` : 'N/A';
            
            responseDiv.innerHTML = `
                <div class="engine-header">
                    <div class="engine-name">${response.engine.toUpperCase()}</div>
                    <div class="engine-rank">${rankDisplay}</div>
                </div>
                <div class="engine-metrics">
                    <div class="engine-metric">
                        <span class="engine-metric-label">Score</span>
                        <span class="engine-metric-value">${scoreDisplay}</span>
                    </div>
                    <div class="engine-metric">
                        <span class="engine-metric-label">Cost</span>
                        <span class="engine-metric-value">${costDisplay}</span>
                    </div>
                    <div class="engine-metric">
                        <span class="engine-metric-label">Tokens</span>
                        <span class="engine-metric-value">${tokensDisplay}</span>
                    </div>
                    <div class="engine-metric">
                        <span class="engine-metric-label">Time</span>
                        <span class="engine-metric-value">${timeDisplay}</span>
                    </div>
                </div>
                <div class="engine-text">${response.text}</div>
            `;
            
            responsesContainer.appendChild(responseDiv);
        });
        
        roundsContainer.appendChild(roundDiv);
    });
}

// Initialize
checkStatus();
setInterval(checkStatus, 5000);
