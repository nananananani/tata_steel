/**
 * Rib Test Module Controller
 */

let currentDiameter = 12;
let currentFile = null;

document.addEventListener('DOMContentLoaded', () => {
    const ribElements = {
        uploadZone: document.getElementById('ribUploadZone'),
        fileInput: document.getElementById('ribFileInput'),
        preview: document.getElementById('ribPreview'),
        previewImage: document.getElementById('ribPreviewImage'),
        analyzeBtn: document.getElementById('ribAnalyzeBtn'),
        resultsContent: document.getElementById('ribResultsContent'),
        resultsVisual: document.getElementById('ribResultsVisual'),
        placeholderData: document.getElementById('ribPlaceholderData'),
        placeholderVisual: document.getElementById('ribPlaceholderVisual'),
        statusText: document.getElementById('ribStatusText'),
        statusReason: document.getElementById('ribStatusReason'),
        statusIcon: document.getElementById('ribStatusIcon'),
        arValueDisplay: document.getElementById('arValueDisplay'),
        parametersContainer: document.getElementById('ribParametersContainer'),
        debugImage: document.getElementById('ribDebugImage')
    };

    // 1. Diameter Selection
    document.querySelectorAll('#ribDiameterOptions .diameter-card').forEach(card => {
        card.onclick = () => {
            document.querySelectorAll('#ribDiameterOptions .diameter-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            currentDiameter = parseFloat(card.dataset.diameter);
        };
    });

    // 2. File Upload
    Common.initFileUpload(ribElements, (file) => {
        Common.openCropper(file, 'ribPreviewImage', (croppedFile) => {
            currentFile = croppedFile;
            ribElements.uploadZone.classList.add('hidden');
            ribElements.preview.classList.remove('hidden');
            ribElements.analyzeBtn.disabled = false;
        });
    });

    // 3. Remove Image
    document.getElementById('ribRemoveImage').onclick = () => {
        currentFile = null;
        ribElements.uploadZone.classList.remove('hidden');
        ribElements.preview.classList.add('hidden');
        ribElements.analyzeBtn.disabled = true;
        ribElements.resultsContent.classList.add('hidden');
        ribElements.resultsVisual.classList.add('hidden');
        ribElements.placeholderData.classList.remove('hidden');
        ribElements.placeholderVisual.classList.remove('hidden');
    };

    // 4. Analyze
    ribElements.analyzeBtn.onclick = () => {
        if (!currentFile) return;

        Common.runAnalysis('/api/rib-test', currentFile, currentDiameter, (data) => {
            displayResults(data, ribElements);
        });
    };
});

function displayResults(data, elements) {
    const isPass = data.status === 'PASS';
    Common.updateStatusUI(elements, isPass, data.reason);

    // AR Value
    if (data.ar_value !== undefined && elements.arValueDisplay) {
        elements.arValueDisplay.textContent = data.ar_value.toFixed(4);
        elements.arValueDisplay.className = `text-5xl font-black ${isPass ? 'text-white' : 'text-red-400'}`;
    }

    // Rib Parameters
    if (elements.parametersContainer) {
        elements.parametersContainer.innerHTML = `
            ${Common.createMetricRow('Detected Ribs', true, data.rib_count)}
            ${Common.createMetricRow('Avg Height', true, data.avg_height_mm + ' mm')}
            ${Common.createMetricRow('Avg Spacing', true, data.avg_spacing_mm + ' mm')}
            ${Common.createMetricRow('Avg Angle', true, data.avg_angle_deg + '°')}
            ${Common.createMetricRow('Avg Length', true, data.avg_length_mm + ' mm')}
        `;
    }

    // Debug Image
    if (data.debug_image_url && elements.debugImage) {
        elements.debugImage.src = data.debug_image_url + '?t=' + Date.now();
    }
}
