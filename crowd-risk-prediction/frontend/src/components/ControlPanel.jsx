import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Button,
  ButtonGroup,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Box,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Divider,
  Chip,
  Tooltip,
  Grid,
  useTheme
} from '@mui/material';
import {
  ExpandMore,
  Speed,
  Settings,
  Analytics,
  Memory,
  Storage,
  NetworkCheck
} from '@mui/icons-material';
import './ControlPanel.css';

const ControlPanel = ({ analysisMode, setAnalysisMode, videoInfo }) => {
  const theme = useTheme();
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [ciriWeights, setCiriWeights] = useState({
    density: 0.15,
    entropy: 0.15,
    foi: 0.20,
    lmcs: 0.20,
    densityGrad: 0.15,
    accel: 0.15
  });
  const [advancedSettings, setAdvancedSettings] = useState({
    confidenceThreshold: 0.7,
    temporalWindow: 8,
    spatialResolution: 32,
    enableRealtime: true,
    enableAlerts: true
  });

  const handleModeChange = (mode) => {
    setAnalysisMode(mode);
  };

  const handleWeightChange = (weightName, value) => {
    setCiriWeights(prev => ({
      ...prev,
      [weightName]: parseFloat(value)
    }));
  };

  const handleAdvancedSettingChange = (settingName, value) => {
    setAdvancedSettings(prev => ({
      ...prev,
      [settingName]: value
    }));
  };

  const analysisModes = [
    { 
      value: 'density', 
      label: 'Density Analysis', 
      description: 'Focus on crowd density patterns',
      icon: '👥'
    },
    { 
      value: 'motion', 
      label: 'Motion Analysis', 
      description: 'Analyze movement patterns and flow',
      icon: '🏃'
    },
    { 
      value: 'full', 
      label: 'Full CIRI Model', 
      description: 'Comprehensive risk assessment',
      icon: '📊'
    }
  ];

  return (
    <Card sx={{ height: '100%', bgcolor: 'background.paper' }}>
      <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Settings sx={{ mr: 1, color: 'primary.main' }} />
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Analysis Control
            </Typography>
          </Box>
          
          <Typography variant="body2" color="text.secondary">
            Configure analysis parameters and system settings
          </Typography>
        </Box>

        <Divider sx={{ mb: 3 }} />

        {/* Analysis Mode Selection */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 500 }}>
            Analysis Mode
          </Typography>
          <ButtonGroup 
            variant="outlined" 
            fullWidth 
            orientation={window.innerWidth < 600 ? 'vertical' : 'horizontal'}
          >
            {analysisModes.map((mode) => (
              <Tooltip key={mode.value} title={mode.description}>
                <Button
                  onClick={() => handleModeChange(mode.value)}
                  variant={analysisMode === mode.value ? 'contained' : 'outlined'}
                  sx={{
                    py: 1.5,
                    px: 2,
                    minHeight: 56,
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 0.5
                  }}
                >
                  <Typography variant="h4">{mode.icon}</Typography>
                  <Typography variant="caption" sx={{ fontWeight: 500 }}>
                    {mode.label}
                  </Typography>
                </Button>
              </Tooltip>
            ))}
          </ButtonGroup>
        </Box>

        <Divider sx={{ mb: 3 }} />

        {/* System Status */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 500 }}>
            System Status
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Chip 
                icon={<Memory />} 
                label="Memory: 68%" 
                color="primary" 
                variant="outlined" 
                size="small"
                sx={{ width: '100%', justifyContent: 'flex-start' }}
              />
            </Grid>
            <Grid item xs={6}>
              <Chip 
                icon={<Speed />} 
                label="FPS: 28.5" 
                color="success" 
                variant="outlined" 
                size="small"
                sx={{ width: '100%', justifyContent: 'flex-start' }}
              />
            </Grid>
            <Grid item xs={6}>
              <Chip 
                icon={<Storage />} 
                label="Storage: 42%" 
                color="warning" 
                variant="outlined" 
                size="small"
                sx={{ width: '100%', justifyContent: 'flex-start' }}
              />
            </Grid>
            <Grid item xs={6}>
              <Chip 
                icon={<NetworkCheck />} 
                label="Network: Good" 
                color="info" 
                variant="outlined" 
                size="small"
                sx={{ width: '100%', justifyContent: 'flex-start' }}
              />
            </Grid>
          </Grid>
        </Box>

        <Divider sx={{ mb: 3 }} />

        {/* Advanced Settings Accordion */}
        <Accordion 
          expanded={showAdvancedOptions} 
          onChange={() => setShowAdvancedOptions(!showAdvancedOptions)}
          sx={{ 
            mt: 'auto',
            bgcolor: 'background.default',
            borderRadius: 2
          }}
        >
          <AccordionSummary
            expandIcon={<ExpandMore />}
            sx={{ minHeight: 48 }}
          >
            <Analytics sx={{ mr: 1, color: 'secondary.main' }} />
            <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
              Advanced Configuration
            </Typography>
          </AccordionSummary>
          
          <AccordionDetails>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              
              {/* CIRI Component Weights */}
              <Box>
                <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 500 }}>
                  CIRI Component Weights
                </Typography>
                
                {Object.entries({
                  density: 'Density (D)',
                  entropy: 'Directional Entropy (H_d)',
                  foi: 'Flow Opposition Index (FOI)',
                  lmcs: 'Local Motion Compression Score (LMCS)',
                  densityGrad: 'Density Gradient (∇D)',
                  accel: 'Acceleration (Δv)'
                }).map(([key, label]) => (
                  <Box key={key} sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">{label}</Typography>
                      <Typography variant="body2" color="primary">
                        {(ciriWeights[key] * 100).toFixed(0)}%
                      </Typography>
                    </Box>
                    <Slider
                      value={ciriWeights[key]}
                      onChange={(e, value) => handleWeightChange(key, value)}
                      step={0.01}
                      min={0}
                      max={1}
                      valueLabelDisplay="auto"
                      sx={{ 
                        color: 'primary.main',
                        '& .MuiSlider-thumb': {
                          backgroundColor: 'primary.main',
                        }
                      }}
                    />
                  </Box>
                ))}
              </Box>

              <Divider />

              {/* Advanced Parameters */}
              <Box>
                <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 500, mb: 2 }}>
                  Advanced Parameters
                </Typography>
                
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Box>
                    <Typography variant="body2" gutterBottom>
                      Confidence Threshold: {advancedSettings.confidenceThreshold}
                    </Typography>
                    <Slider
                      value={advancedSettings.confidenceThreshold}
                      onChange={(e, value) => handleAdvancedSettingChange('confidenceThreshold', value)}
                      step={0.05}
                      min={0.1}
                      max={0.95}
                      valueLabelDisplay="auto"
                    />
                  </Box>
                  
                  <FormControl fullWidth size="small">
                    <InputLabel>Temporal Window</InputLabel>
                    <Select
                      value={advancedSettings.temporalWindow}
                      label="Temporal Window"
                      onChange={(e) => handleAdvancedSettingChange('temporalWindow', e.target.value)}
                    >
                      <MenuItem value={4}>4 Frames</MenuItem>
                      <MenuItem value={8}>8 Frames</MenuItem>
                      <MenuItem value={16}>16 Frames</MenuItem>
                      <MenuItem value={32}>32 Frames</MenuItem>
                    </Select>
                  </FormControl>
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={advancedSettings.enableRealtime}
                        onChange={(e) => handleAdvancedSettingChange('enableRealtime', e.target.checked)}
                        color="primary"
                      />
                    }
                    label="Real-time Processing"
                  />
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={advancedSettings.enableAlerts}
                        onChange={(e) => handleAdvancedSettingChange('enableAlerts', e.target.checked)}
                        color="secondary"
                      />
                    }
                    label="Enable Risk Alerts"
                  />
                </Box>
              </Box>
            </Box>
          </AccordionDetails>
        </Accordion>
      </CardContent>
    </Card>
  );
};

export default ControlPanel;