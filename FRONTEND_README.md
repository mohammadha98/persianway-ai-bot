# Persian Agriculture RAG System - Frontend UI

A modern, responsive web interface for the Persian Agriculture Knowledge Base RAG system built with React and Material-UI.

## 🌟 Features

### 💬 Chat Interface
- Real-time conversational interaction with the RAG system
- Message history with user and bot avatars
- Confidence scores and source information display
- Responsive design for all devices
- Persian text support

### ⚙️ Settings Management
- Knowledge base status monitoring
- Document processing controls
- System information display
- PDF and Excel QA file processing

### 📝 Data Contribution
- Step-by-step form for adding new knowledge
- Tag-based categorization
- Source attribution and references
- Real-time validation and feedback

## 🚀 Quick Start

### Prerequisites

- **Node.js** (v16 or higher) - [Download here](https://nodejs.org/)
- **npm** (comes with Node.js)
- **Python** (v3.8 or higher) for the FastAPI backend

### Option 1: Automated Build (Recommended)

1. **Run the build script:**
   ```bash
   python build_frontend.py
   ```

2. **Start the FastAPI server:**
   ```bash
   python main.py
   ```

3. **Open your browser:**
   - Frontend: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Option 2: Manual Build

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Build the frontend:**
   ```bash
   npm run build
   ```

4. **Start the FastAPI server:**
   ```bash
   cd ..
   python main.py
   ```

## 🏗️ Development Setup

For development with hot reload:

1. **Start the FastAPI backend:**
   ```bash
   python main.py
   ```

2. **In a new terminal, start the React development server:**
   ```bash
   cd frontend
   npm start
   ```

3. **Access the application:**
   - Development Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## 📁 Project Structure

```
frontend/
├── public/
│   ├── index.html          # Main HTML template
│   └── favicon.ico         # App icon
├── src/
│   ├── components/         # React components
│   │   ├── ChatInterface.js    # Chat functionality
│   │   ├── SettingsPanel.js    # Settings management
│   │   └── DataContribution.js # Knowledge contribution
│   ├── services/
│   │   └── api.js          # API service layer
│   ├── App.js              # Main application component
│   ├── index.js            # Application entry point
│   └── index.css           # Global styles
├── package.json            # Dependencies and scripts
└── build/                  # Production build (generated)
```

## 🎨 UI Components

### Chat Interface
- **Message Display**: User and bot messages with avatars
- **Confidence Indicators**: Visual confidence scores
- **Source References**: Clickable source information
- **Input Controls**: Send message and clear chat functionality

### Settings Panel
- **Status Cards**: Knowledge base and processing status
- **Action Buttons**: Document processing controls
- **Information Display**: System statistics and configuration

### Data Contribution
- **Multi-step Form**: Guided knowledge entry process
- **Tag Management**: Visual tag creation and display
- **Validation**: Real-time form validation
- **Preview**: Review before submission

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `frontend/` directory:

```env
# API Configuration
REACT_APP_API_URL=http://localhost:8000/api

# App Configuration
REACT_APP_NAME=Persian Agriculture RAG
REACT_APP_VERSION=1.0.0
```

### API Integration

The frontend communicates with the FastAPI backend through these endpoints:

- **Chat**: `POST /api/chat/`
- **Knowledge Base Query**: `POST /api/knowledge_base/query`
- **Knowledge Base Status**: `GET /api/knowledge_base/status`
- **Contribute Knowledge**: `POST /api/knowledge_base/contribute`
- **Process Documents**: `POST /api/knowledge_base/process`
- **Conversations**: `GET /api/conversations/`

## 🎯 Usage Guide

### Using the Chat Interface

1. **Start a Conversation**: Type your agriculture-related question in Persian or English
2. **View Responses**: See AI-generated answers with confidence scores
3. **Check Sources**: Click on source references for more information
4. **Clear History**: Use the clear button to start fresh

### Managing Settings

1. **Monitor Status**: Check knowledge base and processing status
2. **Process Documents**: Upload and process PDF documents
3. **View Statistics**: See document counts and system information

### Contributing Knowledge

1. **Basic Information**: Enter title, content, and source
2. **Add Tags**: Categorize with relevant keywords
3. **Review**: Check your contribution before submitting
4. **Submit**: Add to the knowledge base

## 🔍 Troubleshooting

### Common Issues

**Frontend not loading:**
- Ensure the build process completed successfully
- Check that `frontend/build/` directory exists
- Verify FastAPI server is running on port 8000

**API connection errors:**
- Confirm backend server is running
- Check CORS configuration in FastAPI
- Verify API endpoints are accessible

**Build failures:**
- Ensure Node.js and npm are installed
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules` and run `npm install` again

### Development Issues

**Hot reload not working:**
- Restart the development server
- Check for syntax errors in console
- Ensure port 3000 is not blocked

**Styling issues:**
- Clear browser cache
- Check Material-UI theme configuration
- Verify CSS imports

## 🧪 Testing

Run the test suite:

```bash
cd frontend
npm test
```

Run tests with coverage:

```bash
npm test -- --coverage
```

## 📦 Building for Production

1. **Optimize build:**
   ```bash
   cd frontend
   npm run build
   ```

2. **Analyze bundle size:**
   ```bash
   npm run build -- --analyze
   ```

## 🌐 Browser Support

- **Chrome** (latest)
- **Firefox** (latest)
- **Safari** (latest)
- **Edge** (latest)
- **Mobile browsers** (iOS Safari, Chrome Mobile)

## 🎨 Customization

### Theming

Modify the Material-UI theme in `src/index.js`:

```javascript
const theme = createTheme({
  palette: {
    primary: {
      main: '#4caf50', // Green
    },
    secondary: {
      main: '#ff9800', // Orange
    },
  },
});
```

### Styling

Custom styles are in `src/index.css`. The application uses:
- Material-UI components
- CSS-in-JS with emotion
- Responsive design principles
- Persian text support

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make changes and test**
4. **Commit changes**: `git commit -m 'Add new feature'`
5. **Push to branch**: `git push origin feature/new-feature`
6. **Create Pull Request**

## 📄 License

This project is part of the Persian Agriculture RAG System. Please refer to the main project license.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the FastAPI backend logs
3. Check browser developer console for errors
4. Create an issue in the project repository

---

**Happy coding! 🌱**