import React, { useRef, useEffect, useMemo } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  useTheme,
  alpha
} from '@mui/material';
import './HeatmapOverlay.css';

const HeatmapOverlay = ({ heatmapData, analysisMode = 'full', isVisible = true }) => {
  const canvasRef = useRef(null);
  const theme = useTheme();

  // Memoize the rendering to optimize performance
  const imageData = useMemo(() => {
    if (!heatmapData || !isVisible) return null;

    const height = heatmapData.length;
    const width = heatmapData[0]?.length || 0;
    
    if (height === 0 || width === 0) return null;

    // Create a temporary canvas to generate image data
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const ctx = tempCanvas.getContext('2d');
    const imageDataObj = ctx.createImageData(width, height);

    // Define color mapping based on analysis mode
    const getColorForValue = (value) => {
      const normalizedValue = Math.min(Math.max(value, 0), 1);

      let r, g, b, a = 200;
      
      if (analysisMode === 'density') {
        // Blue scale for density
        r = 0;
        g = Math.floor(normalizedValue * 200);
        b = Math.floor(55 + normalizedValue * 200);
      } else if (analysisMode === 'motion') {
        // Green scale for motion
        r = Math.floor(normalizedValue * 100);
        g = Math.floor(100 + normalizedValue * 155);
        b = Math.floor(normalizedValue * 100);
      } else {
        // Red-yellow-green scale for full CIRI
        if (normalizedValue < 0.3) {
          r = Math.floor(normalizedValue * 200 / 0.3);
          g = 200;
          b = 0;
        } else if (normalizedValue < 0.6) {
          r = 200 + Math.floor((normalizedValue - 0.3) * 55 / 0.3);
          g = 200;
          b = 0;
        } else {
          r = 255;
          g = Math.floor(200 - (normalizedValue - 0.6) * 200 / 0.4);
          b = 0;
        }
      }

      return [r, g, b, a];
    };

    // Fill image data
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const value = heatmapData[y][x];
        const [r, g, b, a] = getColorForValue(value);
        
        const idx = (y * width + x) * 4;
        imageDataObj.data[idx] = r;
        imageDataObj.data[idx + 1] = g;
        imageDataObj.data[idx + 2] = b;
        imageDataObj.data[idx + 3] = a;
      }
    }

    return imageDataObj;
  }, [heatmapData, analysisMode, isVisible]);

  useEffect(() => {
    if (!canvasRef.current || !imageData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    canvas.width = imageData.width;
    canvas.height = imageData.height;
    ctx.putImageData(imageData, 0, 0);
  }, [imageData]);

  if (!isVisible || !heatmapData) {
    return null;
  }

  return (
    <Box className="heatmap-overlay-container">
      <canvas
        ref={canvasRef}
        className="heatmap-canvas"
        style={{
          opacity: 0.7,
          pointerEvents: 'none',
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%'
        }}
      />
      
      <Card 
        sx={{ 
          position: 'absolute',
          bottom: 20,
          left: 20,
          bgcolor: alpha(theme.palette.background.paper, 0.85),
          backdropFilter: 'blur(10px)',
          borderRadius: 2,
          boxShadow: 3
        }}
      >
        <CardContent sx={{ p: 2 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            Risk Level Legend
          </Typography>
          
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box 
                sx={{ 
                  width: 20, 
                  height: 20, 
                  borderRadius: 1,
                  bgcolor: analysisMode === 'density' ? '#00ff37' : 
                          analysisMode === 'motion' ? '#00aa00' : 'linear-gradient(90deg, #00ff00, #ffff00, #ff7700, #ff0000)'
                }} 
              />
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {analysisMode === 'density' ? 'Density' : 
                 analysisMode === 'motion' ? 'Motion' : 'CIRI Risk'}
              </Typography>
            </Box>
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
              <Box sx={{ display: 'flex', gap: 0.5 }}>
                {['#00ff00', '#ffff00', '#ff7700', '#ff0000'].map((color, index) => (
                  <Box 
                    key={index}
                    sx={{ 
                      width: 15, 
                      height: 15, 
                      bgcolor: color,
                      borderRadius: 0.5
                    }} 
                  />
                ))}
              </Box>
              <Typography variant="caption" color="text.secondary">
                Low to Critical Risk
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default HeatmapOverlay;