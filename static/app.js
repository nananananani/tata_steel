// Ring Test Web Interface - JavaScript
console.log('🚀 Ring Test Interface JavaScript loaded successfully!');

let selectedDiameter = 12;
let selectedFile = null;

// DOM Elements
let uploadZone, fileInput, preview, previewImage, removeImageBtn, analyzeBtn, loadingState;
let resultsContent, resultsVisual, placeholderData, placeholderVisual;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('✅ DOM Content Loaded - Initializing interface...');

    // Initialize DOM elements
    uploadZone = document.getElementById('uploadZone');
    fileInput = document.getElementById('fileInput');
    preview = document.getElementById('preview');
    previewImage = document.getElementById('previewImage');
    removeImageBtn = document.getElementById('removeImage');
    analyzeBtn = document.getElementById('analyzeBtn');
    loadingState = document.getElementById('loadingState');
    resultsContent = document.getElementById('resultsContent');
    resultsVisual = document.getElementById('resultsVisual');
    placeholderData = document.getElementById('placeholderData');
    placeholderVisual = document.getElementById('placeholderVisual');

    console.log('📦 DOM Elements:', {
        uploadZone: !!uploadZone,
        fileInput: !!fileInput,
        analyzeBtn: !!analyzeBtn
    });

    setupDiameterSelection();
    setupFileUpload();
    setupAnalyzeButton();

    console.log('✨ Interface initialized successfully!');
});

// Diameter Selection
function setupDiameterSelection() {
    const diameterCards = document.querySelectorAll('.diameter-card');

    diameterCards.forEach(card => {
        card.addEventListener('click', () => {
            // Remove selected class from all cards
            diameterCards.forEach(c => {
                c.classList.remove('selected', 'border-blue-500');
                c.classList.add('border-gray-600');
            });

            // Add selected class to clicked card
            card.classList.add('selected', 'border-blue-500');
            card.classList.remove('border-gray-600');

            // Update selected diameter
            selectedDiameter = parseInt(card.dataset.diameter);
            console.log('Selected diameter:', selectedDiameter);
        });
    });
}

// File Upload
function setupFileUpload() {
    // Click to upload
    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files[0]);
    });

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        handleFileSelect(e.dataTransfer.files[0]);
    });

    // Remove image
    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearImage();
    });
}

function handleFileSelect(file) {
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
        showNotification('Please select an image file', 'error');
        return;
    }

    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        showNotification('File size must be less than 10MB', 'error');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadZone.classList.add('hidden');
        preview.classList.remove('hidden');
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    selectedFile = null;
    fileInput.value = '';
    uploadZone.classList.remove('hidden');
    preview.classList.add('hidden');
    analyzeBtn.disabled = true;

    resultsContent.classList.add('hidden');
    resultsVisual.classList.add('hidden');
    placeholderData.classList.remove('hidden');
    placeholderVisual.classList.remove('hidden');
}

// Analyze Button
function setupAnalyzeButton() {
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) {
            showNotification('Please select an image first', 'error');
            return;
        }

        await analyzeImage();
    });
}

async function analyzeImage() {
    // Show loading state
    loadingState.classList.remove('hidden');
    resultsContent.classList.add('hidden');
    resultsVisual.classList.add('hidden');
    placeholderData.classList.add('hidden');
    placeholderVisual.classList.add('hidden');
    analyzeBtn.disabled = true;

    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('diameter', selectedDiameter);

        // Send request
        const response = await fetch('/api/ring-test', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        // Hide loading, show results
        loadingState.classList.add('hidden');
        displayResults(result);

    } catch (error) {
        console.error('Error:', error);
        loadingState.classList.add('hidden');
        showNotification('Error analyzing image: ' + error.message, 'error');
        analyzeBtn.disabled = false;
    }
}

function displayResults(result) {
    resultsContent.classList.remove('hidden');
    resultsVisual.classList.remove('hidden');

    // Status Badge
    const statusIcon = document.getElementById('statusIcon');
    const statusText = document.getElementById('statusText');
    const statusReason = document.getElementById('statusReason');

    if (result.status === 'PASS') {
        statusIcon.classList.remove('hidden');
        statusIcon.innerHTML = `
            <svg class="w-8 h-8 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
            </svg>
        `;
        statusIcon.className = 'w-12 h-12 rounded-full flex items-center justify-center bg-green-500/20 mr-6';
        statusText.className = 'text-2xl font-bold text-green-400';
        statusText.textContent = 'PASS';
    } else {
        statusIcon.classList.remove('hidden');
        statusIcon.innerHTML = `
            <svg class="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
        `;
        statusIcon.className = 'w-12 h-12 rounded-full flex items-center justify-center bg-red-500/20 mr-6';
        statusText.className = 'text-2xl font-bold text-red-400';
        statusText.textContent = 'FAIL';
    }

    statusReason.textContent = result.reason;

    // Level 1 Results
    if (result.level1) {
        const level1Container = document.getElementById('level1Results');
        level1Container.innerHTML = `
            <div class="bg-gray-800/50 rounded-xl p-6">
                <div class="space-y-4">
                    ${createStandardItem('Layers Detected', result.level1.regions_visible)}
                    ${createStandardItem('Outer Ring', result.level1.ring_continuous)}
                    ${createStandardItem('Concentricity', result.level1.concentric)}
                    ${createStandardItem('Thickness Uniformity', result.level1.thickness_uniform)}
                </div>
            </div>
        `;
    }

    // Level 2 Results
    if (result.level2) {
        const level2Container = document.getElementById('level2Results');
        level2Container.innerHTML = '';

        const thickness = result.level2.thickness_mm;
        const thicknessStr = thickness.toFixed(2);
        const lowLimit = selectedDiameter * 0.07;
        const highLimit = selectedDiameter * 0.10;

        const isMinMet = thickness >= lowLimit;
        const isMaxMet = thickness <= highLimit;
        const overallPass = isMinMet && isMaxMet;

        level2Container.innerHTML = `
            <div class="bg-gray-800/50 rounded-xl p-6 mb-4">
                <!-- Section 1: Observations -->
                <div class="text-xs text-gray-400 font-bold uppercase tracking-widest mb-4 border-b border-gray-700/50 pb-2">
                    Observations (in mm)
                </div>
                <div class="space-y-4 mb-8">
                    <div class="flex justify-between items-center">
                        <div class="flex items-center">
                            <span class="w-1.5 h-1.5 rounded-full bg-blue-500/50 mr-3"></span>
                            <span class="text-gray-300 text-sm">Diameter of rebar, D</span>
                        </div>
                        <span class="text-sm font-bold text-white">${selectedDiameter} mm</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <div class="flex items-center">
                            <span class="w-1.5 h-1.5 rounded-full bg-blue-500/50 mr-3"></span>
                            <span class="text-gray-300 text-sm">Measured thickness, t<sub>TM</sub></span>
                        </div>
                        <span class="text-sm font-bold text-white">${thicknessStr} mm</span>
                    </div>
                </div>

                <!-- Section 2: Questions/Criteria -->
                <div class="text-xs text-gray-400 font-bold uppercase tracking-widest mb-4 border-b border-gray-700/50 pb-2">
                    L2 Acceptance Criteria
                </div>
                <div class="space-y-4">
                    ${createStandardItem(`Is <b>${thicknessStr}mm</b> ≥ ${lowLimit.toFixed(2)}mm ?`, isMinMet)}
                    ${createStandardItem(`Is <b>${thicknessStr}mm</b> ≤ ${highLimit.toFixed(2)}mm ?`, isMaxMet)}
                </div>

                <!-- Decision -->
                <div class="mt-8 pt-4 border-t border-gray-700">
                    <div class="flex justify-between items-center">
                         <span class="text-xs font-bold text-gray-500 uppercase tracking-widest">Decision</span>
                         <span class="px-3 py-1 rounded text-xs font-bold ${overallPass ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}">
                            ${overallPass ? 'ACCEPT REBAR LOT' : 'REJECT REBAR LOT'}
                         </span>
                    </div>
                </div>
            </div>
        `;
    }

    // Debug Image
    if (result.debug_image_url) {
        const debugImage = document.getElementById('debugImage');
        debugImage.src = result.debug_image_url + '?t=' + Date.now();
    }

    // Re-enable analyze button
    analyzeBtn.disabled = false;
}

/**
 * Creates a standardized result item with bullet and Shadcn-style icon
 */
function createStandardItem(label, passed, customValue = null) {
    const icon = passed ?
        `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="text-green-400"><polyline points="20 6 9 17 4 12"></polyline></svg>` :
        `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="text-red-400"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`;

    const valDisplay = customValue !== null ? customValue : (passed ? 'YES' : 'NO');

    return `
        <div class="flex justify-between items-center">
            <div class="flex items-center">
                <span class="w-1.5 h-1.5 rounded-full bg-gray-500/50 mr-3"></span>
                <span class="text-gray-300 text-sm font-medium">${label}</span>
            </div>
            <div class="flex items-center gap-3">
                <span class="text-sm font-bold ${passed ? 'text-green-400' : 'text-red-400'}">
                    ${valDisplay}
                </span>
                <div class="w-6 h-6 rounded-full flex items-center justify-center ${passed ? 'bg-green-500/10' : 'bg-red-500/10'} border ${passed ? 'border-green-500/20' : 'border-red-500/20'}">
                    ${icon}
                </div>
            </div>
        </div>
    `;
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${type === 'error' ? 'bg-red-500' : 'bg-blue-500'
        } text-white`;
    notification.textContent = message;

    document.body.appendChild(notification);

    // Remove after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}
