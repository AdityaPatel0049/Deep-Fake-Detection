# FraudWatch AI - Frontend

A modern web interface for the AI-powered fraud detection system.

## Features

- **Drag & Drop Upload**: Easy file upload with visual feedback
- **Real-time Analysis**: Live connection to AI models for instant results
- **Comprehensive Results**: Detailed fraud risk assessment with multiple analysis layers
- **Responsive Design**: Works on desktop and mobile devices
- **System Status**: Real-time backend connectivity monitoring

## Quick Start

### Prerequisites

- Backend server running on `http://127.0.0.1:5000`
- Modern web browser with JavaScript enabled

### Running the Frontend

1. **Start the Backend First**:
   ```bash
   cd fraud-detection-ai
   python backend/main.py
   ```

2. **Start the Frontend Server**:
   ```bash
   cd frontend
   python -m http.server 8080
   ```

3. **Open in Browser**:
   - Navigate to `http://localhost:8080`
   - Or open `frontend/index.html` directly in your browser

## Usage

1. **Upload Media**: Drag and drop an image or video file, or click "Browse Files"
2. **Preview**: Review your uploaded file before analysis
3. **Analyze**: Click "Run Analysis" to start the AI inspection
4. **Review Results**: Examine the comprehensive fraud assessment

## Analysis Components

The system provides:

- **AI Ensemble Prediction**: Deep learning analysis using multiple models
- **Metadata Analysis**: EXIF data examination for suspicious patterns
- **Tampering Detection**: Error Level Analysis (ELA) for image manipulation
- **Fraud Risk Score**: Overall probability of fraudulent activity
- **Actionable Recommendations**: Approve, Review, or Reject suggestions

## Supported File Types

- **Images**: JPG, PNG, GIF, BMP (max 50MB)
- **Videos**: MP4, AVI, MOV, MKV (max 50MB)

## System Requirements

- Backend API running on port 5000
- Frontend served on any port (default: 8080)
- CORS enabled for cross-origin requests

## Troubleshooting

### Backend Connection Issues
- Ensure the backend server is running: `python backend/main.py`
- Check that the API is accessible at `http://127.0.0.1:5000/api/models`
- Verify CORS settings in the backend

### File Upload Problems
- Check file size (max 50MB)
- Verify supported file formats
- Ensure stable network connection

### Analysis Errors
- Confirm all Python dependencies are installed
- Check that the Hugging Face model is accessible
- Review backend logs for detailed error messages

## Development

### File Structure
```
frontend/
├── index.html          # Main HTML interface
├── css/
│   └── index.css       # Styling and animations
└── js/
    └── app.js          # Frontend logic and API integration
```

### Customization

- **Styling**: Modify `css/index.css` for visual changes
- **Functionality**: Update `js/app.js` for new features
- **API Integration**: Change `API_URL` constant for different backends

## Browser Compatibility

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Security Notes

- Files are processed server-side only
- No client-side AI processing
- All analysis happens on the backend
- Uploaded files are temporarily stored and cleaned up automatically