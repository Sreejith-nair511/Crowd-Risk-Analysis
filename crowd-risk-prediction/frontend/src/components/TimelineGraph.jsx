import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box,
  Chip,
  useTheme,
  useMediaQuery
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { motion } from 'framer-motion';

const TimelineGraph = ({ riskTimeline, isLoading, currentFrame }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [selectedFrame, setSelectedFrame] = useState(null);

  if (isLoading) {
    return (
      <Box sx={{ 
        height: isMobile ? 250 : 350, 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center' 
      }}>
        <Box sx={{ width: '100%', textAlign: 'center' }}>
          <Typography variant={isMobile ? "body1" : "h6"} gutterBottom>
            Analyzing Risk Patterns...
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Processing video data
          </Typography>
        </Box>
      </Box>
    );
  }

  if (!riskTimeline || riskTimeline.length === 0) {
    return (
      <Card sx={{ 
        height: isMobile ? 250 : 350,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <Typography variant="body1" color="text.secondary">
          No risk data available for visualization
        </Typography>
      </Card>
    );
  }

  // Prepare chart data with enhanced formatting
  const chartData = riskTimeline.map((item, index) => ({
    frame: item.frame,
    ciri_score: item.ciri_score,
    risk_level: item.ciri_score < 0.3 ? 'Low' : 
                item.ciri_score < 0.6 ? 'Medium' : 
                item.ciri_score < 0.8 ? 'High' : 'Critical',
    risk_color: item.ciri_score < 0.3 ? '#4caf50' : 
                item.ciri_score < 0.6 ? '#ff9800' : 
                item.ciri_score < 0.8 ? '#ff5722' : '#f44336'
  }));

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Box 
          sx={{ 
            bgcolor: 'background.paper',
            p: 2,
            borderRadius: 2,
            boxShadow: 3,
            border: `2px solid ${data.risk_color}`
          }}
        >
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            Frame {data.frame}
          </Typography>
          <Typography variant="body2" sx={{ color: data.risk_color, fontWeight: 500 }}>
            Risk Level: {data.risk_level}
          </Typography>
          <Typography variant="body2">
            CIRI Score: {data.ciri_score.toFixed(3)}
          </Typography>
        </Box>
      );
    }
    return null;
  };

  // Calculate statistics
  const stats = {
    maxRisk: riskTimeline && Array.isArray(riskTimeline) ? Math.max(...riskTimeline.map(r => r.ciri_score)) : 0,
    avgRisk: riskTimeline && Array.isArray(riskTimeline) ? riskTimeline.reduce((sum, r) => sum + r.ciri_score, 0) / riskTimeline.length : 0,
    highRiskCount: riskTimeline && Array.isArray(riskTimeline) ? riskTimeline.filter(r => r.ciri_score > 0.7).length : 0,
    criticalRiskCount: riskTimeline && Array.isArray(riskTimeline) ? riskTimeline.filter(r => r.ciri_score > 0.9).length : 0
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card sx={{ 
        height: '100%', 
        bgcolor: 'background.paper',
        boxShadow: 2
      }}>
        <CardContent sx={{ 
          height: '100%', 
          display: 'flex', 
          flexDirection: 'column',
          p: isMobile ? 2 : 3
        }}>
          <Box sx={{ 
            mb: isMobile ? 2 : 3,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 1
          }}>
            <Typography 
              variant={isMobile ? "h6" : "h5"} 
              sx={{ fontWeight: 600 }}
            >
              Risk Timeline Analysis
            </Typography>
            <Chip 
              label={`Avg: ${(stats.avgRisk * 100).toFixed(1)}%`}
              size={isMobile ? "small" : "medium"}
              sx={{
                bgcolor: stats.avgRisk > 0.7 ? theme.palette.error.main : 
                        stats.avgRisk > 0.4 ? theme.palette.warning.main : 
                        theme.palette.success.main,
                color: 'white',
                fontWeight: 600
              }}
            />
          </Box>

          {/* Chart Container */}
          <Box sx={{ 
            flex: 1, 
            minHeight: isMobile ? 200 : 250,
            mb: isMobile ? 2 : 3
          }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={chartData}
                margin={{ 
                  top: 20, 
                  right: isMobile ? 10 : 30, 
                  left: isMobile ? 10 : 20, 
                  bottom: isMobile ? 30 : 50 
                }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
                <XAxis 
                  dataKey="frame" 
                  label={{ 
                    value: 'Frame Number', 
                    position: 'insideBottom', 
                    offset: -10,
                    fontSize: isMobile ? 10 : 12
                  }}
                  tick={{ 
                    fill: theme.palette.text.secondary,
                    fontSize: isMobile ? 10 : 12
                  }}
                />
                <YAxis 
                  domain={[0, 1]}
                  label={{ 
                    value: 'CIRI Score', 
                    angle: -90, 
                    position: 'insideLeft',
                    fontSize: isMobile ? 10 : 12
                  }}
                  tick={{ 
                    fill: theme.palette.text.secondary,
                    fontSize: isMobile ? 10 : 12
                  }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend 
                  wrapperStyle={{
                    fontSize: isMobile ? '10px' : '12px'
                  }}
                />
                
                {/* Risk threshold reference lines */}
                <ReferenceLine 
                  y={0.3} 
                  stroke="#4caf50" 
                  strokeDasharray="3 3" 
                  strokeWidth={1}
                />
                <ReferenceLine 
                  y={0.6} 
                  stroke="#ff9800" 
                  strokeDasharray="3 3" 
                  strokeWidth={1}
                />
                <ReferenceLine 
                  y={0.8} 
                  stroke="#f44336" 
                  strokeDasharray="3 3" 
                  strokeWidth={1}
                />
                
                {/* Current frame indicator */}
                {currentFrame !== undefined && currentFrame < chartData.length && (
                  <ReferenceLine 
                    x={currentFrame} 
                    stroke="#9c27b0" 
                    strokeDasharray="3 3"
                    strokeWidth={2}
                  />
                )}
                
                {/* Main risk line */}
                <Line
                  type="monotone"
                  dataKey="ciri_score"
                  stroke="#2196f3"
                  strokeWidth={isMobile ? 2 : 3}
                  dot={{ r: isMobile ? 3 : 4, fill: '#2196f3' }}
                  activeDot={{ r: isMobile ? 5 : 6, stroke: '#2196f3', strokeWidth: 2 }}
                  name="CIRI Score"
                />
              </LineChart>
            </ResponsiveContainer>
          </Box>

          {/* Statistics Summary */}
          <Box sx={{ 
            display: 'grid', 
            gridTemplateColumns: isMobile ? '1fr 1fr' : 'repeat(auto-fit, minmax(120px, 1fr))', 
            gap: isMobile ? 1 : 2,
            mt: 'auto'
          }}>
            <Card 
              variant="outlined" 
              sx={{ 
                p: isMobile ? 1 : 2,
                textAlign: 'center'
              }}
            >
              <Typography 
                variant={isMobile ? "body2" : "body1"} 
                sx={{ color: theme.palette.text.secondary, mb: 0.5 }}
              >
                Critical Events
              </Typography>
              <Typography 
                variant={isMobile ? "h6" : "h5"} 
                sx={{ 
                  color: theme.palette.error.main, 
                  fontWeight: 700 
                }}
              >
                {stats.criticalRiskCount}
              </Typography>
            </Card>
            
            <Card 
              variant="outlined" 
              sx={{ 
                p: isMobile ? 1 : 2,
                textAlign: 'center'
              }}
            >
              <Typography 
                variant={isMobile ? "body2" : "body1"} 
                sx={{ color: theme.palette.text.secondary, mb: 0.5 }}
              >
                High Risk Events
              </Typography>
              <Typography 
                variant={isMobile ? "h6" : "h5"} 
                sx={{ 
                  color: theme.palette.warning.main, 
                  fontWeight: 700 
                }}
              >
                {stats.highRiskCount}
              </Typography>
            </Card>
            
            <Card 
              variant="outlined" 
              sx={{ 
                p: isMobile ? 1 : 2,
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
                  color: theme.palette.info.main, 
                  fontWeight: 700 
                }}
              >
                {stats.maxRisk.toFixed(3)}
              </Typography>
            </Card>
            
            <Card 
              variant="outlined" 
              sx={{ 
                p: isMobile ? 1 : 2,
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
                  color: theme.palette.success.main, 
                  fontWeight: 700 
                }}
              >
                {stats.avgRisk.toFixed(3)}
              </Typography>
            </Card>
          </Box>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default TimelineGraph;