// Global state
let activeDataset = 'heart_disease';
let activeAlgorithm = 'random_forest';
let datasetMetadata = {};
let trainedModelsCache = {}; // Cache trained metrics: { "dataset_algo": metrics }

// Chart instances
let rocChartInstance = null;
let importanceChartInstance = null;
let classDistChartInstance = null;
let correlationChartInstance = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

async function initApp() {
    setupEventListeners();
    await selectDataset(activeDataset);
}

function setupEventListeners() {
    // Dataset selectors
    const datasetCards = document.querySelectorAll('.dataset-card');
    datasetCards.forEach(card => {
        card.addEventListener('click', async () => {
            datasetCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            
            const selectedDs = card.getAttribute('data-dataset');
            await selectDataset(selectedDs);
        });
    });

    // Algorithm selection
    const algoSelect = document.getElementById('algorithm-select');
    algoSelect.addEventListener('change', (e) => {
        activeAlgorithm = e.target.value;
        updateActiveLabels();
    });

    // Train button
    const trainBtn = document.getElementById('train-btn');
    trainBtn.addEventListener('click', async () => {
        await trainActiveModel(true); // Force re-train
    });

    // Prediction form actions
    const predictBtn = document.getElementById('predict-btn');
    predictBtn.addEventListener('click', (e) => {
        e.preventDefault();
        executeInference();
    });

    const resetBtn = document.getElementById('reset-form-btn');
    resetBtn.addEventListener('click', (e) => {
        e.preventDefault();
        document.getElementById('prediction-form').reset();
        hidePredictionResults();
    });

    // Content navigation tabs
    const tabLinks = document.querySelectorAll('.tab-link');
    tabLinks.forEach(link => {
        link.addEventListener('click', () => {
            tabLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            const targetTabId = link.getAttribute('data-tab');
            const tabPanels = document.querySelectorAll('.tab-panel');
            tabPanels.forEach(panel => panel.classList.remove('active'));
            document.getElementById(targetTabId).classList.add('active');

            // Force charts resize/draw when tabs change to prevent zero-width canvas rendering
            if (targetTabId === 'tab-evaluation') {
                const cacheKey = `${activeDataset}_${activeAlgorithm}`;
                if (trainedModelsCache[cacheKey]) {
                    renderEvaluationMetrics(trainedModelsCache[cacheKey]);
                } else {
                    trainActiveModel(false); // Train silently if not in cache
                }
            } else if (targetTabId === 'tab-eda') {
                loadAndRenderEDA();
            }
        });
    });
}

// Handle switching datasets
async function selectDataset(datasetName) {
    activeDataset = datasetName;
    updateActiveLabels();
    hidePredictionResults();

    try {
        // Fetch metadata
        const response = await fetch(`/api/datasets/${datasetName}/metadata`);
        if (!response.ok) throw new Error("Failed to load metadata");
        datasetMetadata[datasetName] = await response.json();
        
        // Render form
        renderDynamicForm(datasetMetadata[datasetName]);

        // Clear active evaluation charts/cache if it changes
        const activeTab = document.querySelector('.tab-link.active').getAttribute('data-tab');
        if (activeTab === 'tab-evaluation') {
            await trainActiveModel(false); // auto-train new combo
        } else if (activeTab === 'tab-eda') {
            await loadAndRenderEDA();
        }
    } catch (err) {
        console.error("Error setting up dataset: ", err);
        showSystemStatus(false);
    }
}

function updateActiveLabels() {
    const algoNameText = document.getElementById('algorithm-select').options[document.getElementById('algorithm-select').selectedIndex].text.split('(')[0].trim();
    const dsNameText = activeDataset === 'heart_disease' ? 'Heart Disease' : activeDataset === 'diabetes' ? 'Diabetes' : 'Breast Cancer';
    
    document.getElementById('lbl-active-algo').textContent = algoNameText;
    document.getElementById('lbl-active-ds').textContent = dsNameText;
}

// Generate patient inputs dynamically based on features schema
function renderDynamicForm(metadata) {
    const form = document.getElementById('prediction-form');
    form.innerHTML = ''; // Clear previous fields

    metadata.features.forEach(feat => {
        const group = document.createElement('div');
        group.className = 'form-group';

        const label = document.createElement('label');
        label.setAttribute('for', `field-${feat.name}`);
        label.textContent = feat.label;
        group.appendChild(label);

        if (feat.type === 'select') {
            const select = document.createElement('select');
            select.id = `field-${feat.name}`;
            select.name = feat.name;
            select.className = 'form-input';

            feat.options.forEach(opt => {
                const option = document.createElement('option');
                option.value = opt.value;
                option.textContent = opt.label;
                if (opt.value === feat.default) option.selected = true;
                select.appendChild(option);
            });
            group.appendChild(select);
        } else {
            // number inputs
            const input = document.createElement('input');
            input.type = 'number';
            input.id = `field-${feat.name}`;
            input.name = feat.name;
            input.className = 'form-input';
            input.value = feat.default;
            if (feat.min !== undefined) input.min = feat.min;
            if (feat.max !== undefined) input.max = feat.max;
            if (feat.step !== undefined) input.step = feat.step;
            group.appendChild(input);

            const help = document.createElement('span');
            help.className = 'form-help';
            help.textContent = `${feat.help} (Range: ${feat.min} - ${feat.max})`;
            group.appendChild(help);
        }

        form.appendChild(group);
    });
}

// Train ML Model
async function trainActiveModel(forceRetrain = false) {
    const cacheKey = `${activeDataset}_${activeAlgorithm}`;
    
    if (!forceRetrain && trainedModelsCache[cacheKey]) {
        renderEvaluationMetrics(trainedModelsCache[cacheKey]);
        return;
    }

    setTrainLoadingState(true);
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_name: activeDataset,
                algorithm: activeAlgorithm
            })
        });

        if (!response.ok) throw new Error("Failed to train model");
        const data = await response.json();
        
        // Save to cache
        trainedModelsCache[cacheKey] = data;
        
        // Render
        renderEvaluationMetrics(data);
        showSystemStatus(true);
    } catch (err) {
        console.error("Training error:", err);
        alert("Failed to train the selected algorithm. Ensure all server-side python modules are functional.");
        showSystemStatus(false);
    } finally {
        setTrainLoadingState(false);
    }
}

function setTrainLoadingState(isLoading) {
    const btn = document.getElementById('train-btn');
    const spinner = btn.querySelector('.spinner');
    const btnText = btn.querySelector('.btn-text');

    if (isLoading) {
        btn.disabled = true;
        spinner.classList.remove('hidden');
        btnText.textContent = 'Training Model...';
    } else {
        btn.disabled = false;
        spinner.classList.add('hidden');
        btnText.textContent = 'Train & Compile Model';
    }
}

// Render model metrics
function renderEvaluationMetrics(metrics) {
    // Fill metrics texts
    document.getElementById('val-accuracy').textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
    document.getElementById('val-precision').textContent = `${(metrics.precision * 100).toFixed(1)}%`;
    document.getElementById('val-recall').textContent = `${(metrics.recall * 100).toFixed(1)}%`;
    document.getElementById('val-f1').textContent = `${(metrics.f1_score * 100).toFixed(1)}%`;
    document.getElementById('val-auc').textContent = metrics.roc_auc.toFixed(3);

    // Update progress bars
    document.getElementById('bar-accuracy').style.width = `${metrics.accuracy * 100}%`;
    document.getElementById('bar-precision').style.width = `${metrics.precision * 100}%`;
    document.getElementById('bar-recall').style.width = `${metrics.recall * 100}%`;
    document.getElementById('bar-f1').style.width = `${metrics.f1_score * 100}%`;

    // Fill Confusion Matrix
    document.getElementById('cm-tn').innerHTML = `${metrics.confusion_matrix.tn}<span class="cm-sub">True Negative</span>`;
    document.getElementById('cm-fp').innerHTML = `${metrics.confusion_matrix.fp}<span class="cm-sub">False Positive</span>`;
    document.getElementById('cm-fn').innerHTML = `${metrics.confusion_matrix.fn}<span class="cm-sub">False Negative</span>`;
    document.getElementById('cm-tp').innerHTML = `${metrics.confusion_matrix.tp}<span class="cm-sub">True Positive</span>`;

    // Draw ROC Chart
    renderROCChart(metrics.roc_curve.fpr, metrics.roc_curve.tpr);

    // Draw Feature Importance Chart
    renderImportanceChart(metrics.feature_importances);
}

// Draw ROC AUC Chart
function renderROCChart(fpr, tpr) {
    const ctx = document.getElementById('rocChart').getContext('2d');
    
    if (rocChartInstance) {
        rocChartInstance.destroy();
    }

    rocChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: fpr.map(v => v.toFixed(2)),
            datasets: [
                {
                    label: 'Model ROC',
                    data: tpr,
                    borderColor: '#00f2fe',
                    backgroundColor: 'rgba(0, 242, 254, 0.05)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.2,
                    pointRadius: 3,
                    pointBackgroundColor: '#00f2fe'
                },
                {
                    label: 'Random Guess',
                    data: fpr, // diagonal y = x
                    borderColor: 'rgba(255, 255, 255, 0.15)',
                    borderDash: [5, 5],
                    borderWidth: 1,
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    title: { display: true, text: 'False Positive Rate', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#94a3b8', maxTicksLimit: 10 }
                },
                y: {
                    title: { display: true, text: 'True Positive Rate', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#94a3b8' },
                    min: 0,
                    max: 1.05
                }
            }
        }
    });
}

// Draw Feature Importance Chart
function renderImportanceChart(importances) {
    const ctx = document.getElementById('importanceChart').getContext('2d');
    
    if (importanceChartInstance) {
        importanceChartInstance.destroy();
    }

    // Limit to top 8 features for cleaner visualization
    const topImportances = importances.slice(0, 8);

    importanceChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: topImportances.map(i => i.feature),
            datasets: [{
                label: 'Importance Score',
                data: topImportances.map(i => i.importance),
                backgroundColor: 'rgba(168, 85, 247, 0.4)',
                borderColor: '#a855f7',
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { size: 11 } }
                }
            }
        }
    });
}

// Load and render EDA Statistics
async function loadAndRenderEDA() {
    try {
        const response = await fetch(`/api/datasets/${activeDataset}/eda`);
        if (!response.ok) throw new Error("Failed to load EDA stats");
        const data = await response.json();

        // 1. Target distribution Pie/Doughnut Chart
        const classDistCtx = document.getElementById('classDistChart').getContext('2d');
        if (classDistChartInstance) {
            classDistChartInstance.destroy();
        }
        
        const labels = Object.keys(data.target_distribution);
        const counts = Object.values(data.target_distribution);

        classDistChartInstance = new Chart(classDistCtx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: counts,
                    backgroundColor: ['rgba(16, 185, 129, 0.5)', 'rgba(239, 68, 68, 0.5)'],
                    borderColor: ['#10b981', '#ef4444'],
                    borderWidth: 1.5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#94a3b8', font: { size: 10 } }
                    }
                },
                cutout: '65%'
            }
        });

        // 2. Correlation Chart (horizontal/vertical bar)
        const corrCtx = document.getElementById('correlationChart').getContext('2d');
        if (correlationChartInstance) {
            correlationChartInstance.destroy();
        }

        const topCorrelations = data.correlations.slice(0, 10); // top 10

        correlationChartInstance = new Chart(corrCtx, {
            type: 'bar',
            data: {
                labels: topCorrelations.map(c => c.feature),
                datasets: [{
                    label: 'Pearson Correlation with Target',
                    data: topCorrelations.map(c => c.correlation),
                    backgroundColor: topCorrelations.map(c => c.correlation >= 0 ? 'rgba(0, 242, 254, 0.4)' : 'rgba(239, 68, 68, 0.4)'),
                    borderColor: topCorrelations.map(c => c.correlation >= 0 ? '#00f2fe' : '#ef4444'),
                    borderWidth: 1.2,
                    borderRadius: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8', font: { size: 10 } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });

        // 3. Populate EDA text statistics summary
        const textContainer = document.getElementById('eda-summary-text');
        textContainer.innerHTML = '';

        const heading = document.createElement('h3');
        heading.textContent = activeDataset === 'heart_disease' ? 'Heart Disease Parameters' : activeDataset === 'diabetes' ? 'Diabetes Parameters' : 'Breast Cancer Parameters';
        textContainer.appendChild(heading);

        const list = document.createElement('div');
        list.className = 'stats-list';

        // Add 4 key stats
        const totalSamples = counts.reduce((a, b) => a + b, 0);
        const diseasePercent = ((counts[1] / totalSamples) * 100).toFixed(1);

        const rows = [
            { lbl: "Total Sample Size", val: totalSamples },
            { lbl: "Normal / Negative Cases", val: counts[0] },
            { lbl: "Disease / Positive Cases", val: counts[1] },
            { lbl: "Disease Prevalence", val: `${diseasePercent}%` }
        ];

        rows.forEach(r => {
            const row = document.createElement('div');
            row.className = 'stats-row';
            row.innerHTML = `<span class="lbl">${r.lbl}</span><span class="val highlight">${r.val}</span>`;
            list.appendChild(row);
        });

        textContainer.appendChild(list);

    } catch (err) {
        console.error("EDA rendering error: ", err);
    }
}

// Run Predict Inference
async function executeInference() {
    const form = document.getElementById('prediction-form');
    
    // Gather inputs
    const inputs = {};
    const formData = new FormData(form);
    
    // Validate inputs
    let hasEmpty = false;
    datasetMetadata[activeDataset].features.forEach(feat => {
        const val = formData.get(feat.name);
        if (val === null || val === '') {
            hasEmpty = true;
            return;
        }
        inputs[feat.name] = Number(val);
    });

    if (hasEmpty) {
        alert("Please specify values for all biometric parameters.");
        return;
    }

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_name: activeDataset,
                algorithm: activeAlgorithm,
                inputs: inputs
            })
        });

        if (!response.ok) throw new Error("Prediction API error");
        const result = await response.json();
        
        displayPredictionResults(result, inputs);
        showSystemStatus(true);
    } catch (err) {
        console.error("Prediction error:", err);
        alert("Failed to execute prediction. Make sure the model has been trained.");
        showSystemStatus(false);
    }
}

function displayPredictionResults(result, inputs) {
    document.getElementById('result-placeholder').classList.add('hidden');
    const display = document.getElementById('result-display');
    display.classList.remove('hidden');

    // Update risk dial probability percentage
    const percentage = Math.round(result.probability * 100);
    document.getElementById('risk-percent').textContent = `${percentage}%`;

    // SVG stroke dashoffset animation (radius of circle is 40, circumference is 2 * PI * r = 251.2)
    const strokeDash = 251.2;
    const offset = strokeDash - (strokeDash * result.probability);
    const fillCircle = document.getElementById('gauge-fill');
    
    // Set color based on risk severity
    let alertColor = '#10b981'; // Green
    if (result.probability >= 0.7) {
        alertColor = '#ef4444'; // Red
    } else if (result.probability >= 0.35) {
        alertColor = '#f59e0b'; // Orange/Yellow
    }
    
    fillCircle.style.stroke = alertColor;
    fillCircle.style.strokeDashoffset = offset;

    // Update Assessment Card details
    const assessBox = document.getElementById('assessment-box');
    const assessTitle = document.getElementById('assessment-title');
    const assessDetails = document.getElementById('assessment-details');

    assessBox.className = 'assessment-box'; // reset

    if (result.disease_detected) {
        assessBox.classList.add('positive');
        assessTitle.textContent = 'POSITIVE DIAGNOSIS RISK DETECTED';
        assessDetails.textContent = `The model assesses high likelihood of positive pathology with a confidence of ${percentage}%. Clinical consultation recommended.`;
    } else {
        assessBox.classList.add('negative');
        assessTitle.textContent = 'NEGATIVE DIAGNOSIS (LOW RISK)';
        assessDetails.textContent = `The model assesses low diagnostic threat levels (confidence score of ${100 - percentage}% negative).`;
    }

    // Populate factors overview list
    const list = document.getElementById('factor-list');
    list.innerHTML = '';

    Object.entries(inputs).forEach(([key, val]) => {
        const item = document.createElement('div');
        item.className = 'factor-item';
        
        // Find label
        const featSchema = datasetMetadata[activeDataset].features.find(f => f.name === key);
        const label = featSchema ? featSchema.label : key;
        
        // Handle categories display
        let valText = val;
        if (featSchema && featSchema.type === 'select') {
            const opt = featSchema.options.find(o => o.value == val);
            if (opt) valText = opt.label;
        }

        item.innerHTML = `<span class="factor-name">${label}</span><span class="factor-val">${valText}</span>`;
        list.appendChild(item);
    });
}

function hidePredictionResults() {
    document.getElementById('result-placeholder').classList.remove('hidden');
    document.getElementById('result-display').classList.add('hidden');
}

function showSystemStatus(isOnline) {
    const statusBox = document.getElementById('connection-status');
    const indicator = statusBox.querySelector('.status-indicator');
    const text = statusBox.querySelector('.status-text');

    if (isOnline) {
        indicator.className = 'status-indicator online';
        text.textContent = 'Connected to Engine';
    } else {
        indicator.className = 'status-indicator offline';
        text.textContent = 'Engine Offline';
    }
}
