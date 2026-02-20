import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Button,
  LinearProgress,
  Alert,
  Card,
  CardContent,
  Stepper,
  Step,
  StepLabel,
  useTheme,
  useMediaQuery,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Grid
} from '@mui/material';
import {
  CloudUpload,
  VideoFile,
  Analytics,
  CheckCircle,
  Error,
  Info,
  Storage,
  Speed,
  Security
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import './UploadPage.css';

const steps = [
  'Upload Video',
  'Processing',
  'Analysis Complete'
];

const UploadPage = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [videoId, setVideoId] = useState('');
  const [error, setError] = useState('');
  const [activeStep, setActiveStep] = useState(0);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (!selectedFile.type.startsWith('video/')) {
        setError('Please select a video file (MP4, AVI, MOV, etc.)');
        return;
      }
      
      setFile(selectedFile);
      setError('');
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a video file first');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setError('');
    setActiveStep(1);

    // Simulate upload progress
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 95) {
          clearInterval(progressInterval);
          return 95;
        }
        return prev + Math.random() * 15;
      });
    }, 200);

    try {
      // Simulate upload success
      setTimeout(() => {
        clearInterval(progressInterval);
        setUploadProgress(100);
        setUploadSuccess(true);
        setVideoId('vid_' + Date.now().toString(36));
        setIsUploading(false);
        setActiveStep(2);
        
        // Show success message
        setTimeout(() => {
          alert(`Analysis started for video: ${videoId}\nCheck the dashboard for results.`);
        }, 500);
      }, 2000);
      
    } catch (err) {
      setError(err.message || 'An error occurred during upload');
      setIsUploading(false);
      setActiveStep(0);
    }
  };

  const handleReset = () => {
    setFile(null);
    setUploadProgress(0);
    setIsUploading(false);
    setUploadSuccess(false);
    setVideoId('');
    setError('');
    setActiveStep(0);
  };

  return (
    <Box sx={{ 
      minHeight: '100vh', 
      bgcolor: 'background.default',
      py: 4
    }}>
      <Container maxWidth="md">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Paper 
            sx={{ 
              p: 4, 
              textAlign: 'center',
              bgcolor: 'background.paper',
              borderRadius: 3,
              boxShadow: 3
            }}
          >
            <Box sx={{ mb: 4 }}>
              <CloudUpload sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
              <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
                Crowd Risk Analysis
              </Typography>
              <Typography variant="h6" color="text.secondary" sx={{ mb: 3 }}>
                Upload your video for advanced crowd instability prediction
              </Typography>
              
              <Stepper 
                activeStep={activeStep} 
                alternativeLabel 
                sx={{ mb: 4 }}
              >
                {steps.map((label) => (
                  <Step key={label}>
                    <StepLabel>{label}</StepLabel>
                  </Step>
                ))}
              </Stepper>
            </Box>

            <Box sx={{ 
              border: '2px dashed', 
              borderColor: 'divider',
              borderRadius: 2,
              p: 4,
              mb: 3,
              bgcolor: 'background.default',
              transition: 'all 0.3s ease',
              '&:hover': {
                borderColor: 'primary.main',
                bgcolor: 'action.hover'
              }
            }}>
              <input
                type="file"
                id="file-input"
                accept="video/*"
                onChange={handleFileChange}
                disabled={isUploading}
                style={{ display: 'none' }}
              />
              <label htmlFor="file-input">
                <Box sx={{ cursor: 'pointer' }}>
                  <VideoFile sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                  <Typography variant="h6" sx={{ mb: 1 }}>
                    {file ? file.name : 'Click to select a video file'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Supports MP4, AVI, MOV, and other common video formats
                  </Typography>
                </Box>
              </label>
            </Box>

            {error && (
              <Alert 
                severity="error" 
                sx={{ mb: 3, borderRadius: 2 }}
                onClose={() => setError('')}
              >
                {error}
              </Alert>
            )}
            
            {file && !isUploading && (
              <Card sx={{ mb: 3, bgcolor: 'background.default' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                    File Information
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemIcon>
                        <Info color="primary" />
                      </ListItemIcon>
                      <ListItemText 
                        primary="File Name" 
                        secondary={file.name} 
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Storage color="primary" />
                      </ListItemIcon>
                      <ListItemText 
                        primary="File Size" 
                        secondary={`${(file.size / (1024 * 1024)).toFixed(2)} MB`} 
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Speed color="primary" />
                      </ListItemIcon>
                      <ListItemText 
                        primary="File Type" 
                        secondary={file.type} 
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon>
                        <Security color="primary" />
                      </ListItemIcon>
                      <ListItemText 
                        primary="Security" 
                        secondary="File will be processed securely and deleted after analysis" 
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            )}

            {isUploading && (
              <Box sx={{ mb: 3 }}>
                <Typography variant="body1" gutterBottom>
                  Uploading and processing your video...
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={uploadProgress} 
                  sx={{ 
                    height: 10, 
                    borderRadius: 5,
                    mb: 2
                  }} 
                />
                <Typography variant="body2" color="text.secondary">
                  {Math.round(uploadProgress)}% complete
                </Typography>
              </Box>
            )}

            {uploadSuccess && (
              <Alert 
                severity="success" 
                icon={<CheckCircle />}
                sx={{ mb: 3, borderRadius: 2 }}
              >
                <Typography variant="h6" sx={{ mb: 1 }}>
                  Upload Successful!
                </Typography>
                <Typography variant="body2">
                  Video ID: <strong>{videoId}</strong>
                </Typography>
              </Alert>
            )}

            <Box sx={{ 
              display: 'flex', 
              gap: 2, 
              justifyContent: 'center',
              flexWrap: 'wrap'
            }}>
              <Button 
                variant="contained" 
                size="large"
                onClick={handleUpload}
                disabled={!file || isUploading}
                startIcon={<CloudUpload />}
                sx={{ 
                  px: 4, 
                  py: 1.5,
                  borderRadius: 2,
                  fontWeight: 600
                }}
              >
                {isUploading ? 'Processing...' : 'Upload & Analyze'}
              </Button>
              
              {(uploadSuccess || isUploading) && (
                <Button 
                  variant="outlined" 
                  size="large"
                  onClick={handleReset}
                  sx={{ 
                    px: 4, 
                    py: 1.5,
                    borderRadius: 2,
                    fontWeight: 600
                  }}
                >
                  Reset
                </Button>
              )}
            </Box>

            <Divider sx={{ my: 4 }} />

            <Box>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
                How It Works
              </Typography>
              
              <Grid container spacing={3} sx={{ mt: 2 }}>
                <Grid item xs={12} md={4}>
                  <Card sx={{ height: '100%', bgcolor: 'background.default' }}>
                    <CardContent>
                      <Analytics sx={{ fontSize: 40, color: 'primary.main', mb: 2 }} />
                      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                        Advanced Analysis
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Our system uses CSRNet for density estimation, optical flow for motion analysis, and spatio-temporal transformers for pattern recognition.
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Card sx={{ height: '100%', bgcolor: 'background.default' }}>
                    <CardContent>
                      <Speed sx={{ fontSize: 40, color: 'secondary.main', mb: 2 }} />
                      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                        Real-time Processing
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Process videos in real-time with optimized inference targeting &lt;100ms per frame for responsive risk assessment.
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Card sx={{ height: '100%', bgcolor: 'background.default' }}>
                    <CardContent>
                      <Security sx={{ fontSize: 40, color: 'success.main', mb: 2 }} />
                      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                        Enterprise Security
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Military-grade encryption, secure processing, and automatic file cleanup ensure your data remains protected.
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Box>
          </Paper>
        </motion.div>
      </Container>
    </Box>
  );
};

export default UploadPage;