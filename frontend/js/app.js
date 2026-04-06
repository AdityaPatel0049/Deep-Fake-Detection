document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const videoPreview = document.getElementById('video-preview');
    const clearBtn = document.getElementById('clear-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    const loadingState = document.getElementById('loading-state');
    const resultsSection = document.getElementById('results-section');
    
    // API CONFIG
    const API_URL = 'http://127.0.0.1:5000/api/predict';
    const HEALTH_URL = 'http://127.0.0.1:5000/api/models';

    let currentFile = null;

    // Check system status on load
    checkSystemStatus();
    testBackendConnection();

    // Check system status periodically
    setInterval(checkSystemStatus, 30000); // Check every 30 seconds

    async function checkSystemStatus() {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.system-status');
        
        try {
            const response = await fetch(HEALTH_URL);
            if (response.ok) {
                statusDot.style.backgroundColor = 'var(--color-success)';
                statusText.innerHTML = '<span class="status-dot"></span> System Online';
                statusText.querySelector('.status-dot').style.backgroundColor = 'var(--color-success)';
            } else {
                throw new Error('Server responded with error');
            }
        } catch (error) {
            statusDot.style.backgroundColor = 'var(--color-danger)';
            statusText.innerHTML = '<span class="status-dot"></span> System Offline';
            statusText.querySelector('.status-dot').style.backgroundColor = 'var(--color-danger)';
        }
    }

    async function testBackendConnection() {
        try {
            console.log('Testing backend connection...');
            const response = await fetch(HEALTH_URL);
            if (response.ok) {
                console.log('✅ Backend connection successful');
            } else {
                console.log('❌ Backend responded with error:', response.status);
            }
        } catch (error) {
            console.log('❌ Backend connection failed:', error.message);
        }
    }

    // --- Drag and Drop Handlers ---
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        let dt = e.dataTransfer;
        let files = dt.files;
        handleFiles(files);
    }
    
    // --- File Input Handlers ---
    
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        
        // Basic validation
        const validImageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp'];
        const validVideoTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv'];
        
        if (!validImageTypes.includes(file.type) && !validVideoTypes.includes(file.type)) {
            alert('Unsupported file type. Please upload an image (JPG, PNG, GIF, BMP) or video (MP4, AVI, MOV, MKV).');
            return;
        }

        // File size validation (50MB limit)
        const maxSize = 50 * 1024 * 1024; // 50MB
        if (file.size > maxSize) {
            alert(`File too large. Maximum size is 50MB. Your file is ${(file.size / (1024 * 1024)).toFixed(1)}MB.`);
            return;
        }

        currentFile = file;
        
        // Setup previes
        const fileReader = new FileReader();
        fileReader.onload = (e) => {
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            
            if (validImageTypes.includes(file.type)) {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
                videoPreview.classList.add('hidden');
            } else {
                videoPreview.src = URL.createObjectURL(file);
                videoPreview.classList.remove('hidden');
                imagePreview.classList.add('hidden');
            }

            // Update file info
            document.getElementById('file-name').textContent = file.name;
            document.getElementById('file-size').textContent = `${(file.size / (1024 * 1024)).toFixed(1)} MB`;
            
            // Hide old results if doing a new upload
            resultsSection.classList.add('hidden');
        };
        fileReader.readAsDataURL(file);
    }

    // --- Action Button Handlers ---
    
    clearBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        resultsSection.classList.add('hidden');
        loadingState.classList.add('hidden');
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Updates
        previewContainer.classList.add('hidden');
        loadingState.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        const startTime = Date.now();
        const formData = new FormData();
        formData.append('media', currentFile);

        try {
            console.log('Starting analysis request to:', API_URL);
            console.log('File to upload:', currentFile.name, currentFile.size, 'bytes');
            
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });

            console.log('Response status:', response.status);
            console.log('Response headers:', Object.fromEntries(response.headers.entries()));

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                console.error('Error response:', errorData);
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }

            const data = await response.json();
            console.log('Success response:', data);
            
            if (data.status !== 'success') {
                throw new Error(data.error || 'Analysis failed');
            }

            const analysisTime = ((Date.now() - startTime) / 1000).toFixed(1);
            populateResults(data, analysisTime);
            
            // Hide loading, show results
            loadingState.classList.add('hidden');
            previewContainer.classList.remove('hidden'); // allow seeing image while viewing results
            resultsSection.classList.remove('hidden');
            

        } catch (error) {
            console.error('Analysis error:', error);
            let errorMessage = error.message;
            
            // Provide more helpful error messages
            if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                errorMessage = 'Cannot connect to the backend server. Please make sure the backend is running on http://127.0.0.1:5000';
            } else if (error.message.includes('CORS')) {
                errorMessage = 'CORS error: Backend server is not allowing requests from this frontend.';
            } else if (error.message.includes('404')) {
                errorMessage = 'API endpoint not found. Please check the backend routes.';
            }
            
            alert('Analysis Error: ' + errorMessage + '\n\nCheck the browser console for more details.');
            loadingState.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        }
    });

    // --- Populate DOM with Results ---
    
    function populateResults(data, analysisTime) {
        // Banner updates
        const banner = document.getElementById('verdict-banner');
        document.getElementById('final-risk-level').textContent = `Risk Level: ${data.risk_level}`;
        document.getElementById('final-recommendation').textContent = `Recommendation: ${data.recommendation}`;
        document.getElementById('final-score').textContent = `${data.fraud_score}%`;
        document.getElementById('analysis-time').textContent = `${analysisTime}s`;
        
        // Reset classes
        banner.className = 'verdict-banner glass-panel';
        if(data.risk_level === 'HIGH' || data.risk_level === 'CRITICAL') banner.classList.add('risk-high');
        else if (data.risk_level === 'MEDIUM') banner.classList.add('risk-medium');
        else banner.classList.add('risk-low');

        // AI Block
        const isFake = data.predictions.predicted_class === 0; // 0 is Fake, 1 is Real
        document.getElementById('ai-prediction').textContent = isFake ? 'AI Generated / Fake' : 'Real Content';
        document.getElementById('ai-confidence').textContent = `${(data.predictions.confidence * 100).toFixed(1)}%`;
        
        const aiProb = data.predictions.probabilities.fake * 100;
        document.getElementById('ai-progress-bar').style.width = `${aiProb}%`;
        document.getElementById('ai-progress-bar').style.backgroundColor = isFake ? 'var(--color-danger)' : 'var(--color-success)';

        // Metadata Flags
        const metaList = document.getElementById('metadata-flags');
        metaList.innerHTML = '';
        data.metadata_analysis.flags.forEach(flag => {
            const li = document.createElement('li');
            li.textContent = flag;
            
            // simple visual heuristic
            if(flag.toLowerCase().includes('missing') || flag.toLowerCase().includes('generate') || flag.toLowerCase().includes('edit')) {
                li.classList.add('flag-warning');
            } else if (flag.toLowerCase().includes('normal')) {
                li.classList.add('flag-success');
            }
            metaList.appendChild(li);
        });

        // Tampering Flags
        const tampList = document.getElementById('tampering-flags');
        tampList.innerHTML = '';
        data.tampering_analysis.flags.forEach(flag => {
            const li = document.createElement('li');
            li.textContent = flag;
            
            if(flag.toLowerCase().includes('high')) {
                li.classList.add('flag-danger');
            } else if (flag.toLowerCase().includes('normal') || flag.toLowerCase().includes('bypassed')) {
                li.classList.add('flag-success');
            }
            tampList.appendChild(li);
        });
    }

});
