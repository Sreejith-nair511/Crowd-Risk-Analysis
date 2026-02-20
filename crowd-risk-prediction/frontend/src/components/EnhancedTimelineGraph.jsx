import React, { useState, useEffect, useRef } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box,
  Chip,
  useTheme,
  useMediaQuery,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  AreaChart,
  Area
} from 'recharts';
import { 
  ZoomIn, 
  ZoomOut, 
  Refresh,
  Assessment
} from '@mui/icons-material';
import { motion } from 'framer-motion';

const EnhancedTimelineGraph = ({ data = [], onFrameSelect, isLoading, currentFrame }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [zoomLevel, setZoomLevel] = useState(1);
  const [chartData, setChartData] = useState([]);
  const chartRef = useRef();

  // Generate sample data if none provided
  useEffect(() => {
    let generatedData = [];
    
    if (data && data.risk_timeline && Array.isArray(data.risk_timeline)) {
      // Handle structured data with risk_timeline
      generatedData = data.risk_timeline.map((item, index) => ({
        frame: item.frame || index,
        riskScore: item.ciri_score || item.riskScore || Math.random() * 0.8 + 0.1,
        density: item.density || Math.random() * 0.7 + 0.2,
        flow: item.flow || Math.random() * 0.6 + 0.1,
        timestamp: item.timestamp || new Date(Date.now() - (data.risk_timeline.length - 1 - index) * 500).toLocaleTimeString()
      }));
    } else if (Array.isArray(data)) {
      // Handle array data directly
      generatedData = data.map((item, index) => ({
        frame: item.frame || index,
        riskScore: item.ciri_score || item.riskScore || Math.random() * 0.8 + 0.1,
        density: item.density || Math.random() * 0.7 + 0.2,
        flow: item.flow || Math.random() * 0.6 + 0.1,
        timestamp: item.timestamp || new Date(Date.now() - (data.length - 1 - index) * 500).toLocaleTimeString()
      }));
    } else {
      // Generate sample data
      generatedData = Array.from({ length: 200 }, (_, i) => ({
        frame: i,
        riskScore: Math.random() * 0.8 + 0.1,
        density: Math.random() * 0.7 + 0.2,
        flow: Math.random() * 0.6 + 0.1,
        timestamp: new Date(Date.now() - (199 - i) * 500).toLocaleTimeString()
      }));
    }
    
    setChartData(generatedData);
  }, [data]);

  const handleFrameClick = (data) => {
    if (data && data.activePayload && data.activePayload[0]) {
      const frameData = data.activePayload[0].payload;
      if (onFrameSelect) {
        onFrameSelect(frameData.frame);
      }
    }
  };

  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev + 0.2, 3));
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev - 0.2, 0.5));
  };

  const handleReset = () => {
    setZoomLevel(1);
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <Card sx={{ 
          bgcolor: theme.palette.background.paper,
          border: `1px solid ${theme.palette.divider}`,
          boxShadow: 4,
          minWidth: 200
        }}>
          <CardContent sx={{ p: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600, color: theme.palette.text.primary }}>
              Frame {label}
            </Typography>
            {payload.map((entry, index) => (
              <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                <Box 
                  sx={{ 
                    width: 10, 
                    height: 10, 
                    borderRadius: '50%', 
                    bgcolor: entry.color,
                    mr: 1 
                  }} 
                />
                <Typography 
                  variant="body2" 
                  sx={{ 
                    color: theme.palette.text.secondary,
                    fontWeight: 500
                  }}
                >
                  {entry.dataKey}: {typeof entry.value === 'number' ? entry.value.toFixed(3) : entry.value}
                </Typography>
              </Box>
            ))}
          </CardContent>
        </Card>
      );
    }
    return null;
  };

  // Calculate statistics
  const riskScores = chartData.map(d => d.riskScore);
  const avgRisk = riskScores.length > 0 ? riskScores.reduce((a, b) => a + b, 0) / riskScores.length : 0;
  const maxRisk = riskScores.length > 0 ? Math.max(...riskScores) : 0;
  const minRisk = riskScores.length > 0 ? Math.min(...riskScores) : 0;

  // Determine risk level
  const getRiskLevel = (risk) => {
    if (risk > 0.7) return { level: 'Critical', color: 'error', text: 'High Risk' };
    if (risk > 0.4) return { level: 'High', color: 'warning', text: 'Medium Risk' };
    return { level: 'Low', color: 'success', text: 'Low Risk' };
  };

  const currentRisk = getRiskLevel(avgRisk);

  if (isLoading) {
    return (
      <Card sx={{ 
        height: isMobile ? 300 : 400,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <Box sx={{ textAlign: 'center' }}>
          <Typography variant="h6" gutterBottom>
            Analyzing Risk Patterns...
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Processing video data in real-time
          </Typography>
        </Box>
      </Card>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        bgcolor: 'background.paper'
      }}>
        <CardContent sx={{ 
          flex: 1, 
          p: isMobile ? 2 : 3,
          pb: isMobile ? 2 : 3
        }}>
          {/* Header with controls */}
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            mb: isMobile ? 2 : 3,
            flexWrap: 'wrap',
            gap: 1
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Assessment sx={{ color: theme.palette.primary.main }} />
              <Typography 
                variant={isMobile ? "h6" : "h5"} 
                sx={{ fontWeight: 600 }}
              >
                Risk Analysis Timeline
              </Typography>
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip 
                label={`${currentRisk.level} Risk`}
                color={currentRisk.color}
                size={isMobile ? "small" : "medium"}
                sx={{ fontWeight: 600 }}
              />
              <Tooltip title="Zoom In">
                <IconButton 
                  size="small" 
                  onClick={handleZoomIn}
                  sx={{ 
                    bgcolor: 'background.default',
                    '&:hover': { bgcolor: 'action.hover' }
                  }}
                >
                  <ZoomIn fontSize={isMobile ? "small" : "medium"} />
                </IconButton>
              </Tooltip>
              <Tooltip title="Zoom Out">
                <IconButton 
                  size="small" 
                  onClick={handleZoomOut}
                  sx={{ 
                    bgcolor: 'background.default',
                    '&:hover': { bgcolor: 'action.hover' }
                  }}
                >
                  <ZoomOut fontSize={isMobile ? "small" : "medium"} />
                </IconButton>
              </Tooltip>
              <Tooltip title="Reset View">
                <IconButton 
                  size="small" 
                  onClick={handleReset}
                  sx={{ 
                    bgcolor: 'background.default',
                    '&:hover': { bgcolor: 'action.hover' }
                  }}
                >
                  <Refresh fontSize={isMobile ? "small" : "medium"} />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          {/* Chart Container */}
          <Box sx={{ 
            height: isMobile ? 250 : 350, 
            mb: isMobile ? 2 : 3,
            position: 'relative'
          }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={chartData}
                onClick={handleFrameClick}
                margin={{ 
                  top: 20, 
                  right: isMobile ? 10 : 30, 
                  left: isMobile ? 10 : 20, 
                  bottom: isMobile ? 30 : 50 
                }}
              >
                <defs>
                  <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={theme.palette.error.main} stopOpacity={0.3}/>
                    <stop offset="50%" stopColor={theme.palette.warning.main} stopOpacity={0.2}/>
                    <stop offset="95%" stopColor={theme.palette.success.main} stopOpacity={0.1}/>
                  </linearGradient>
                  <linearGradient id="densityGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={theme.palette.info.main} stopOpacity={0.3}/>
                    <stop offset="95%" stopColor={theme.palette.info.main} stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                
                <CartesianGrid 
                  strokeDasharray="3 3" 
                  stroke={theme.palette.divider} 
                  vertical={false}
                />
                <XAxis 
                  dataKey="frame" 
                  stroke={theme.palette.text.secondary}
                  tick={{ 
                    fill: theme.palette.text.secondary,
                    fontSize: isMobile ? 10 : 12
                  }}
                  label={{ 
                    value: 'Frame Number', 
                    position: 'insideBottom', 
                    offset: -10,
                    fill: theme.palette.text.secondary
                  }}
                />
                <YAxis 
                  stroke={theme.palette.text.secondary}
                  tick={{ 
                    fill: theme.palette.text.secondary,
                    fontSize: isMobile ? 10 : 12
                  }}
                  domain={[0, 1]}
                  label={{ 
                    value: 'Risk Score', 
                    angle: -90, 
                    position: 'insideLeft',
                    fill: theme.palette.text.secondary
                  }}
                />
                <RechartsTooltip content={<CustomTooltip />} />
                <Legend 
                  verticalAlign="top" 
                  height={36}
                  wrapperStyle={{
                    fontSize: isMobile ? '12px' : '14px'
                  }}
                />
                
                {/* Risk threshold reference lines */}
                <ReferenceLine 
                  y={0.7} 
                  stroke={theme.palette.error.main} 
                  strokeDasharray="3 3"
                  strokeWidth={1}
                >
                  <label 
                    value="High Risk" 
                    position="top" 
                    fill={theme.palette.error.main}
                    fontSize={isMobile ? 10 : 12}
                  />
                </ReferenceLine>
                <ReferenceLine 
                  y={0.4} 
                  stroke={theme.palette.warning.main} 
                  strokeDasharray="3 3"
                  strokeWidth={1}
                >
                  <label 
                    value="Medium Risk" 
                    position="top" 
                    fill={theme.palette.warning.main}
                    fontSize={isMobile ? 10 : 12}
                  />
                </ReferenceLine>
                
                {/* Current frame indicator */}
                {currentFrame !== undefined && currentFrame < chartData.length && (
                  <ReferenceLine 
                    x={currentFrame} 
                    stroke={theme.palette.secondary.main} 
                    strokeDasharray="3 3"
                    strokeWidth={2}
                  >
                    <label 
                      value="Current Frame" 
                      position="top" 
                      fill={theme.palette.secondary.main}
                      fontSize={isMobile ? 10 : 12}
                    />
                  </ReferenceLine>
                )}
                
                {/* Data series */}
                <Area
                  type="monotone"
                  dataKey="riskScore"
                  name="Risk Score"
                  stroke={theme.palette.error.main}
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#riskGradient)"
                  activeDot={{ 
                    r: 6, 
                    stroke: theme.palette.error.main, 
                    strokeWidth: 2,
                    fill: theme.palette.background.paper
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="density"
                  name="Crowd Density"
                  stroke={theme.palette.info.main}
                  strokeWidth={2}
                  dot={{ r: 3, fill: theme.palette.info.main }}
                  strokeDasharray="5 5"
                />
                <Line
                  type="monotone"
                  dataKey="flow"
                  name="Movement Flow"
                  stroke={theme.palette.success.main}
                  strokeWidth={2}
                  dot={{ r: 2, fill: theme.palette.success.main }}
                  strokeDasharray="3 3"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Box>

          {/* Statistics Summary */}
          <Box sx={{ 
            display: 'grid', 
            gridTemplateColumns: isMobile ? '1fr 1fr' : 'repeat(auto-fit, minmax(140px, 1fr))', 
            gap: isMobile ? 1 : 2
          }}>
            <Card 
              variant="outlined" 
              sx={{ 
                p: isMobile ? 1.5 : 2,
                textAlign: 'center'
              }}
            >
              <Typography 
                variant={isMobile ? "body2" : "body1"} 
                sx={{ color: theme.palette.text.secondary, mb: 0.5 }}
              >
                Peak Risk
              </Typography>
              <Typography 
                variant={isMobile ? "h6" : "h5"} 
                sx={{ 
                  color: theme.palette.error.main, 
                  fontWeight: 700 
                }}
              >
                {(maxRisk * 100).toFixed(1)}%
              </Typography>
            </Card>
            
            <Card 
              variant="outlined" 
              sx={{ 
                p: isMobile ? 1.5 : 2,
                textAlign: 'center'
              }}
            >
              <Typography 
                variant={isMobile ? "body2" : "body1"} 
                sx={{ color: theme.palette.text.secondary, mb: 0.5 }}
              >
                Average Risk
              </Typography>
              <Typography 
                variant={isMobile ? "h6" : "h5"} 
                sx={{ 
                  color: theme.palette[currentRisk.color].main, 
                  fontWeight: 700 
                }}
              >
                {(avgRisk * 100).toFixed(1)}%
              </Typography>
            </Card>
            
            <Card 
              variant="outlined" 
              sx={{ 
                p: isMobile ? 1.5 : 2,
                textAlign: 'center'
              }}
            >
              <Typography 
                variant={isMobile ? "body2" : "body1"} 
                sx={{ color: theme.palette.text.secondary, mb: 0.5 }}
              >
                Risk Range
              </Typography>
              <Typography 
                variant={isMobile ? "h6" : "h5"} 
                sx={{ 
                  color: theme.palette.warning.main, 
                  fontWeight: 700 
                }}
              >
                {((maxRisk - minRisk) * 100).toFixed(1)}%
              </Typography>
            </Card>
            
            <Card 
              variant="outlined" 
              sx={{ 
                p: isMobile ? 1.5 : 2,
                textAlign: 'center'
              }}
            >
              <Typography 
                variant={isMobile ? "body2" : "body1"} 
                sx={{ color: theme.palette.text.secondary, mb: 0.5 }}
              >
                Total Frames
              </Typography>
              <Typography 
                variant={isMobile ? "h6" : "h5"} 
                sx={{ 
                  color: theme.palette.info.main, 
                  fontWeight: 700 
                }}
              >
                {chartData.length}
              </Typography>
            </Card>
          </Box>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default EnhancedTimelineGraph;