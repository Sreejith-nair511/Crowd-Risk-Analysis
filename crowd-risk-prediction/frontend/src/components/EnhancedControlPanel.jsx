import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  LinearProgress,
  useTheme,
  Divider
} from '@mui/material';
import {
  ExpandMore,
  Settings,
  Memory,
  Speed,
  Analytics,
  Storage,
  NetworkCheck
} from '@mui/icons-material';
import { motion } from 'framer-motion';

const EnhancedControlPanel = ({ systemStats, onSettingsChange }) => {
  const theme = useTheme();
  const [settings, setSettings] = useState({
    analysisMode: 'full',
    confidenceThreshold: 0.7,
    processingSpeed: 'normal',
    modelPrecision: 'high'
  });

  const handleSettingChange = (key, value) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
    if (onSettingsChange) {
      onSettingsChange(newSettings);
    }
  };

  const systemPerformance = [
    { label: 'CPU Usage', value: systemStats?.cpu || 45, icon: <Memory />, color: 'primary' },
    { label: 'Memory', value: systemStats?.memory || 68, icon: <Storage />, color: 'secondary' },
    { label: 'GPU Utilization', value: systemStats?.gpu || 32, icon: <NetworkCheck />, color: 'info' },
    { label: 'Processing Speed', value: systemStats?.speed || 85, icon: <Speed />, color: 'success' }
  ];

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card sx={{ 
        height: '100%',
        bgcolor: theme.palette.background.paper,
        boxShadow: '0 8px 32px rgba(0,0,0,0.2)'
      }}>
        <CardContent sx={{ p: 0 }}>
          {/* Header */}
          <Box sx={{ 
            p: 3, 
            borderBottom: `1px solid ${theme.palette.divider}`,
            bgcolor: theme.palette.background.default
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
              <Box 
                sx={{ 
                  width: 48, 
                  height: 48, 
                  bgcolor: theme.palette.primary.main,
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                <Settings sx={{ color: 'white', fontSize: 28 }} />
              </Box>
              <Box>
                <Typography variant="h6" sx={{ fontWeight: 700 }}>
                  System Control
                </Typography>
                <Typography variant="body2" sx={{ color: theme.palette.text.secondary }}>
                  Configuration & Monitoring
                </Typography>
              </Box>
            </Box>
            
            {/* System Status Chips */}
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Chip 
                label="Online" 
                color="success" 
                size="small"
                sx={{ fontWeight: 600 }}
              />
              <Chip 
                label="Models Loaded" 
                color="info" 
                size="small"
                sx={{ fontWeight: 600 }}
              />
              <Chip 
                label="Real-time" 
                color="primary" 
                size="small"
                sx={{ fontWeight: 600 }}
              />
            </Box>
          </Box>

          <Box sx={{ p: 3 }}>
            {/* System Performance */}
            <Box sx={{ mb: 4 }}>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 3, display: 'flex', alignItems: 'center', gap: 1 }}>
                <Analytics sx={{ color: theme.palette.primary.main }} />
                System Performance
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {systemPerformance.map((metric, index) => (
                  <Box key={index}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {metric.icon}
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {metric.label}
                        </Typography>
                      </Box>
                      <Typography variant="body2" sx={{ fontWeight: 600, color: theme.palette[metric.color].main }}>
                        {metric.value}%
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={metric.value}
                      sx={{ 
                        height: 8, 
                        borderRadius: 4,
                        bgcolor: theme.palette.background.default,
                        '& .MuiLinearProgress-bar': {
                          borderRadius: 4,
                          bgcolor: theme.palette[metric.color].main
                        }
                      }}
                    />
                  </Box>
                ))}
              </Box>
            </Box>

            <Divider sx={{ my: 3 }} />

            {/* Advanced Settings */}
            <Accordion sx={{ 
              boxShadow: 'none',
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: 2,
              '&:before': { display: 'none' }
            }}>
              <AccordionSummary
                expandIcon={<ExpandMore />}
                sx={{ 
                  bgcolor: theme.palette.background.default,
                  borderRadius: 2,
                  '&.Mui-expanded': { borderRadius: '8px 8px 0 0' }
                }}
              >
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Advanced Configuration
                </Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ pt: 2 }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  
                  {/* Analysis Mode */}
                  <FormControl fullWidth size="small">
                    <InputLabel>Analysis Mode</InputLabel>
                    <Select
                      value={settings.analysisMode}
                      label="Analysis Mode"
                      onChange={(e) => handleSettingChange('analysisMode', e.target.value)}
                      sx={{ borderRadius: 2 }}
                    >
                      <MenuItem value="quick">Quick Analysis</MenuItem>
                      <MenuItem value="balanced">Balanced</MenuItem>
                      <MenuItem value="full">Full Analysis</MenuItem>
                      <MenuItem value="research">Research Mode</MenuItem>
                    </Select>
                  </FormControl>

                  {/* Confidence Threshold */}
                  <Box>
                    <Typography variant="body2" sx={{ fontWeight: 500, mb: 2 }}>
                      Confidence Threshold: {Math.round(settings.confidenceThreshold * 100)}%
                    </Typography>
                    <Slider
                      value={settings.confidenceThreshold}
                      onChange={(e, value) => handleSettingChange('confidenceThreshold', value)}
                      step={0.05}
                      min={0.1}
                      max={0.95}
                      valueLabelDisplay="auto"
                      sx={{ 
                        color: theme.palette.primary.main,
                        '& .MuiSlider-thumb': {
                          borderRadius: '50%',
                          boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
                        }
                      }}
                    />
                  </Box>

                  {/* Processing Speed */}
                  <FormControl fullWidth size="small">
                    <InputLabel>Processing Speed</InputLabel>
                    <Select
                      value={settings.processingSpeed}
                      label="Processing Speed"
                      onChange={(e) => handleSettingChange('processingSpeed', e.target.value)}
                      sx={{ borderRadius: 2 }}
                    >
                      <MenuItem value="fast">Fast (Lower Accuracy)</MenuItem>
                      <MenuItem value="normal">Normal</MenuItem>
                      <MenuItem value="precise">Precise (Slower)</MenuItem>
                    </Select>
                  </FormControl>

                  {/* Model Precision */}
                  <FormControl fullWidth size="small">
                    <InputLabel>Model Precision</InputLabel>
                    <Select
                      value={settings.modelPrecision}
                      label="Model Precision"
                      onChange={(e) => handleSettingChange('modelPrecision', e.target.value)}
                      sx={{ borderRadius: 2 }}
                    >
                      <MenuItem value="low">Low Precision</MenuItem>
                      <MenuItem value="medium">Medium Precision</MenuItem>
                      <MenuItem value="high">High Precision</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
              </AccordionDetails>
            </Accordion>
          </Box>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default EnhancedControlPanel;