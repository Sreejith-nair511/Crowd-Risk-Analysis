// API service for crowd risk prediction
const API_BASE_URL = 'http://localhost:8000';

class ApiService {
  async uploadVideo(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Upload error:', error);
      throw error;
    }
  }

  async analyzeVideo(videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/analyze/${videoId}`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Analysis error:', error);
      throw error;
    }
  }

  async getAnalysisResults(videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/results/${videoId}`, {
        method: 'GET',
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Results error:', error);
      throw error;
    }
  }

  async getSystemStatus() {
    try {
      const response = await fetch(`${API_BASE_URL}/`, {
        method: 'GET',
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Status check error:', error);
      throw error;
    }
  }

  // Mock data for demonstration when API is not available
  getMockAnalysisResults() {
    return {
      video_id: 'mock_video_' + Date.now(),
      frames_analyzed: 100,
      average_risk: 0.45,
      peak_risk: 0.82,
      risk_timeline: Array.from({ length: 100 }, (_, i) => ({
        frame: i,
        ciri_score: Math.random() * 0.8 + 0.1,
        density: Math.random() * 0.7 + 0.2,
        flow: Math.random() * 0.6 + 0.1,
        timestamp: new Date(Date.now() - (99 - i) * 1000).toISOString()
      })),
      processing_time: 15.2,
      model_accuracy: 0.94
    };
  }
}

// Create singleton instance
const api = new ApiService();

export default api;