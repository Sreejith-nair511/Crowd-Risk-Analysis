import React, { useState, useRef, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Grid,
  Card,
  CardContent,
  Box,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip,
  Divider,
  Paper,
  useTheme,
  useMediaQuery
} from '@mui/material';
import {
  VideoLibrary,
  Assessment,
  Timeline,
  Warning,
  CheckCircle,
  Error,
  Info,
  Settings,
  Refresh,
  CloudUpload,
  Speed,
  Memory
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import VideoPlayer from '../components/VideoPlayer';
import HeatmapOverlay from '../components/HeatmapOverlay';
import TimelineGraph from '../components/TimelineGraph';
import EnhancedTimelineGraph from '../components/EnhancedTimelineGraph';
import ControlPanel from '../components/ControlPanel';
import EnhancedControlPanel from '../components/EnhancedControlPanel';
import AdvancedAnalytics from '../components/AdvancedAnalytics';
import './Dashboard.css';

const Dashboard = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  const [currentFrame, setCurrentFrame] = useState(0);
  const [riskHeatmap, setRiskHeatmap] = useState(null);
  const [analysisMode, setAnalysisMode] = useState('full');
  const [videoInfo, setVideoInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(() => {
    // Load from localStorage if available
    const savedResults = localStorage.getItem('analysisResults');
    if (savedResults) {
      try {
        const parsed = JSON.parse(savedResults);
        // Ensure risk_timeline is an array
        if (parsed.risk_timeline && Array.isArray(parsed.risk_timeline)) {
          return parsed;
        }
        // If it's just an array of results
        if (Array.isArray(parsed)) {
          return { risk_timeline: parsed };
        }
      } catch (e) {
        console.error('Error parsing saved results:', e);
      }
    }
    return null;
  });
  const [showAdvancedAnalytics, setShowAdvancedAnalytics] = useState(false);
  const [systemStats, setSystemStats] = useState({
    cpu: 45,
    memory: 68,
    gpu: 32,
    fps: 28.5
  });

  // Generate mock data for demonstration
  const mockRiskTimeline = Array.from({ length: 200 }, (_, i) => ({
    frame: i,
    ciri_score: Math.random() * 0.9 + 0.05,
    risk_level: Math.random() > 0.8 ? 'high' : Math.random() > 0.5 ? 'medium' : 'low'
  }));

  useEffect(() => {
    // Simulate loading analysis results
    setIsLoading(true);
    setTimeout(() => {
      setAnalysisResults(mockRiskTimeline);
      setIsLoading(false);
    }, 1500);
  }, []);

  const handleFrameChange = (frameNumber) => {
    setCurrentFrame(frameNumber);
    
    if (analysisResults) {
      const frameResult = analysisResults[frameNumber] || analysisResults[0];
      const riskValue = frameResult ? frameResult.ciri_score : 0.5;
      
      const mockHeatmap = generateMockHeatmap(480, 640, riskValue);
      setRiskHeatmap(mockHeatmap);
    }
  };

  const generateMockHeatmap = (height, width, riskLevel) => {
    const heatmap = [];
    for (let i = 0; i < height; i++) {
      const row = [];
      for (let j = 0; j < width; j++) {
        let value = Math.random() * 0.2;
        
        if (Math.random() < riskLevel * 0.15) {
          const centerX = width / 2 + (Math.random() - 0.5) * width / 3;
          const centerY = height / 2 + (Math.random() - 0.5) * height / 3;
          const dist = Math.sqrt(Math.pow(j - centerX, 2) + Math.pow(i - centerY, 2));
          
          if (dist < 60) {
            value = riskLevel * (1 - dist / 60);
          }
        }
        
        row.push(Math.min(1.0, value));
      }
      heatmap.push(row);
    }
    return heatmap;
  };

  const getRiskLevelColor = (riskLevel) => {
    switch (riskLevel) {
      case 'high': return theme.palette.error.main;
      case 'medium': return theme.palette.warning.main;
      case 'low': return theme.palette.success.main;
      default: return theme.palette.info.main;
    }
  };

  const getRiskLevelLabel = (riskLevel) => {
    switch (riskLevel) {
      case 'high': return 'High Risk';
      case 'medium': return 'Medium Risk';
      case 'low': return 'Low Risk';
      default: return 'Unknown';
    }
  };

  return (
    <Box sx={{ 
      minHeight: '100vh', 
      bgcolor: 'background.default',
      color: 'text.primary'
    }}>
      {/* Header */}
      <AppBar 
        position="static" 
        sx={{ 
          boxShadow: 3,
          bgcolor: 'background.paper',
          borderBottom: `1px solid ${theme.palette.divider}`
        }}
      >
        <Toolbar>
          <VideoLibrary sx={{ mr: 2, color: 'primary.main' }} />
          <Typography 
            variant="h6" 
            component="div" 
            sx={{ 
              flexGrow: 1, 
              fontWeight: 600,
              color: 'text.primary'
            }}
          >
            Crowd Risk Intelligence Platform
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Tooltip title="System Performance">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Speed fontSize="small" />
                <Typography variant="body2">{systemStats.fps.toFixed(1)} FPS</Typography>
              </Box>
            </Tooltip>
            
            <Tooltip title="Memory Usage">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Memory fontSize="small" />
                <Typography variant="body2">{systemStats.memory}%</Typography>
              </Box>
            </Tooltip>
            
            <IconButton color="primary">
              <Refresh />
            </IconButton>
            
            <IconButton color="primary" component="a" href="/upload">
              <CloudUpload />
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ py: 3 }}>
        {/* Status Chips */}
        <Box sx={{ mb: 3 }}>
          <Grid container spacing={2}>
            <Grid item>
              <Chip 
                icon={<CheckCircle />} 
                label="System Online" 
                color="success" 
                variant="outlined"
              />
            </Grid>
            <Grid item>
              <Chip 
                icon={<Info />} 
                label="Analysis Ready" 
                color="info" 
                variant="outlined"
              />
            </Grid>
            <Grid item>
              <Chip 
                icon={<Settings />} 
                label="Model Loaded" 
                color="primary" 
                variant="outlined"
              />
            </Grid>
          </Grid>
        </Box>

        {/* Main Dashboard Content */}
        <Grid container spacing={3}>
          {/* Video Analysis Section */}
          <Grid item xs={12} lg={8}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Card sx={{ height: '100%' }}>
                <CardContent sx={{ p: 0 }}>
                  <Box sx={{ p: 3, borderBottom: `1px solid ${theme.palette.divider}` }}>
                    <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
                      Real-time Video Analysis
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Live crowd monitoring with CIRI-based risk assessment
                    </Typography>
                  </Box>
                  
                  <Box sx={{ p: 3 }}>
                    <VideoPlayer 
                      onFrameChange={handleFrameChange}
                      currentFrame={currentFrame}
                      videoSrc="/sample-video.mp4"
                    />
                    
                    {riskHeatmap && (
                      <Box sx={{ mt: 2 }}>
                        <HeatmapOverlay 
                          heatmapData={riskHeatmap} 
                          analysisMode={analysisMode}
                          isVisible={true}
                        />
                      </Box>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>

          {/* Control Panel Section */}
          <Grid item xs={12} lg={4}>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <EnhancedControlPanel 
                systemStats={systemStats}
                onSettingsChange={(settings) => console.log('Settings:', settings)}
              />
            </motion.div>
          </Grid>

          {/* Timeline and Analytics */}
          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                    <Timeline sx={{ mr: 2, color: 'primary.main' }} />
                    <Typography variant="h5" sx={{ fontWeight: 600 }}>
                      Risk Timeline Analysis
                    </Typography>
                  </Box>
                  
                  <EnhancedTimelineGraph 
                    data={analysisResults || []}
                    isLoading={isLoading}
                    currentFrame={currentFrame}
                    onFrameSelect={(frame) => setCurrentFrame(frame)}
                  />
                </CardContent>
              </Card>
            </motion.div>
          </Grid>

          {/* Summary Statistics */}
          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <Card>
                <CardContent>
                  <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                    Risk Intelligence Summary
                  </Typography>
                  
                  {isLoading ? (
                    <Box sx={{ width: '100%' }}>
                      <LinearProgress />
                    </Box>
                  ) : (
                    <Grid container spacing={3}>
                      <Grid item xs={12} sm={6} md={3}>
                        <Paper 
                          sx={{ 
                            p: 3, 
                            textAlign: 'center',
                            bgcolor: 'background.paper',
                            border: `2px solid ${theme.palette.error.main}`,
                            borderRadius: 2
                          }}
                        >
                          <Error sx={{ fontSize: 40, color: 'error.main', mb: 1 }} />
                          <Typography variant="h4" sx={{ fontWeight: 700, color: 'error.main' }}>
                            {analysisResults?.risk_timeline ? 
                              analysisResults.risk_timeline.filter(r => r.ciri_score > 0.8).length : 
                              analysisResults ? 
                                (Array.isArray(analysisResults) ? 
                                  analysisResults.filter(r => r.ciri_score > 0.8).length : 0) : 0}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Critical Risk Events
                          </Typography>
                        </Paper>
                      </Grid>
                      
                      <Grid item xs={12} sm={6} md={3}>
                        <Paper 
                          sx={{ 
                            p: 3, 
                            textAlign: 'center',
                            bgcolor: 'background.paper',
                            border: `2px solid ${theme.palette.warning.main}`,
                            borderRadius: 2
                          }}
                        >
                          <Warning sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
                          <Typography variant="h4" sx={{ fontWeight: 700, color: 'warning.main' }}>
                            {analysisResults?.risk_timeline ? 
                              analysisResults.risk_timeline.filter(r => r.ciri_score > 0.6 && r.ciri_score <= 0.8).length : 
                              analysisResults ? 
                                (Array.isArray(analysisResults) ? 
                                  analysisResults.filter(r => r.ciri_score > 0.6 && r.ciri_score <= 0.8).length : 0) : 0}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            High Risk Events
                          </Typography>
                        </Paper>
                      </Grid>
                      
                      <Grid item xs={12} sm={6} md={3}>
                        <Paper 
                          sx={{ 
                            p: 3, 
                            textAlign: 'center',
                            bgcolor: 'background.paper',
                            border: `2px solid ${theme.palette.info.main}`,
                            borderRadius: 2
                          }}
                        >
                          <Assessment sx={{ fontSize: 40, color: 'info.main', mb: 1 }} />
                          <Typography variant="h4" sx={{ fontWeight: 700, color: 'info.main' }}>
                            {analysisResults?.risk_timeline ? 
                              (Math.max(...analysisResults.risk_timeline.map(r => r.ciri_score)).toFixed(2)) : 
                              analysisResults ? 
                                (Array.isArray(analysisResults) ? 
                                  (Math.max(...analysisResults.map(r => r.ciri_score)).toFixed(2)) : '0.00') : '0.00'}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Peak Risk Score
                          </Typography>
                        </Paper>
                      </Grid>
                      
                      <Grid item xs={12} sm={6} md={3}>
                        <Paper 
                          sx={{ 
                            p: 3, 
                            textAlign: 'center',
                            bgcolor: 'background.paper',
                            border: `2px solid ${theme.palette.success.main}`,
                            borderRadius: 2
                          }}
                        >
                          <CheckCircle sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
                          <Typography variant="h4" sx={{ fontWeight: 700, color: 'success.main' }}>
                            {analysisResults?.risk_timeline ? 
                              (analysisResults.risk_timeline.reduce((sum, r) => sum + r.ciri_score, 0) / analysisResults.risk_timeline.length).toFixed(2) : 
                              analysisResults ? 
                                (Array.isArray(analysisResults) ? 
                                  (analysisResults.reduce((sum, r) => sum + r.ciri_score, 0) / analysisResults.length).toFixed(2) : '0.00') : '0.00'}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Average Risk Score
                          </Typography>
                        </Paper>
                      </Grid>
                    </Grid>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
          
          {/* Advanced Analytics Toggle */}
          <Grid item xs={12}>
            <Box sx={{ textAlign: 'center', my: 3 }}>
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Paper 
                  onClick={() => setShowAdvancedAnalytics(!showAdvancedAnalytics)}
                  sx={{ 
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 2,
                    p: 2,
                    cursor: 'pointer',
                    bgcolor: 'background.paper',
                    border: `2px solid ${theme.palette.primary.main}`,
                    borderRadius: 3,
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      boxShadow: 3,
                      transform: 'translateY(-2px)'
                    }
                  }}
                >
                  <Assessment sx={{ fontSize: 28, color: 'primary.main' }} />
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    {showAdvancedAnalytics ? 'Hide' : 'Show'} Advanced Analytics
                  </Typography>
                  <Settings sx={{ fontSize: 20, color: 'text.secondary' }} />
                </Paper>
              </motion.div>
            </Box>
          </Grid>

          {/* Advanced Analytics Section */}
          {showAdvancedAnalytics && (
            <Grid item xs={12}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <AdvancedAnalytics />
              </motion.div>
            </Grid>
          )}
        </Grid>
      </Container>
    </Box>
  );
};

export default Dashboard;