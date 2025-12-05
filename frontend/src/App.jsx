import { useState } from 'react'
import './App.css'

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async () => {
    if (!text.trim()) return
    
    setLoading(true)
    setError(null)
    setResult(null)

    console.log(text);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      })

      if (!response.ok) {
        throw new Error('Failed to get prediction')
      }

      const data = await response.json()
      setResult(data.prediction)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setText('')
    setResult(null)
    setError(null)
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>Fake News Detector</h1>
        <p>AI-powered verification tool</p>
      </header>

      <main className="main-content">
        <div className="input-section">
          <textarea
            className="news-input"
            placeholder="Paste the news article here to verify..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
          />
          <div className="button-group">
            <button 
              className="analyze-btn" 
              onClick={handleSubmit}
              disabled={loading || !text.trim()}
            >
              {loading ? 'Analyzing...' : 'Verify News'}
            </button>
            <button className="clear-btn" onClick={handleClear}>
              Clear
            </button>
          </div>
        </div>

        {error && <div className="error-message">{error}</div>}

        {result && (
          <div className={`result-card ${result.toLowerCase()}`}>
            <h2>Prediction Result</h2>
            <div className="result-badge">
              {result}
            </div>
            <p className="result-description">
              {result === 'REAL' 
                ? 'This article appears to be credible based on our analysis.'
                : 'Caution: This article shows signs of being fake news.'}
            </p>
          </div>
        )}
      </main>
      
      <footer className="footer">
        <p>Powered by Machine Learning &bull; React &bull; FastAPI</p>
      </footer>
    </div>
  )
}

export default App