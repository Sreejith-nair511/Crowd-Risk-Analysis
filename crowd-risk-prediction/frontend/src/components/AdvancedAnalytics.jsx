import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Tabs,
  Tab,
  useTheme
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ScatterChart,
  Scatter,
  ZAxis
} from 'recharts';
import { motion } from 'framer-motion';

const AdvancedAnalytics = ({ analyticsData }) => {
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);

  // Generate comprehensive sample data
  const generateSampleData = () => {
    const timeData = Array.from({ length: 24 }, (_, i) => ({
      hour: `${i}:00`,
      crowd_density: Math.random() * 800 + 200,
      risk_score: Math.random() * 0.8 + 0.1,
      flow_complexity: Math.random() * 0.7 + 0.2,
      lmcs: Math.random() * 0.6 + 0.1
    }));

    const riskDistribution = [
      { name: 'Low Risk', value: 45, color: '#4caf50' },
      { name: 'Medium Risk', value: 30, color: '#ff9800' },
      { name: 'High Risk', value: 20, color: '#f44336' },
      { name: 'Critical Risk', value: 5, color: '#9c27b0' }
    ];

    const featureImportance = [
      { feature: 'Density', importance: 0.35, fullMark: 1 },
      { feature: 'Optical Flow', importance: 0.25, fullMark: 1 },
      { feature: 'Heat Distribution', importance: 0.20, fullMark: 1 },
      { feature: 'Flow Complexity', importance: 0.15, fullMark: 1 },
      { feature: 'LMCS', importance: 0.05, fullMark: 1 }
    ];

    const spatialData = Array.from({ length: 50 }, () => ({
      x: Math.random() * 100,
      y: Math.random() * 100,
      risk: Math.random() * 0.9 + 0.1,
      density: Math.random() * 800 + 100
    }));

    return { timeData, riskDistribution, featureImportance, spatialData };
  };

  const data = analyticsData || generateSampleData();

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <Card sx={{ 
          bgcolor: theme.palette.background.paper,
          border: `1px solid ${theme.palette.divider}`,
          boxShadow: 4,
          p: 2
        }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            {label}
          </Typography>
          {payload.map((entry, index) => (
            <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
              <Box 
                sx={{ 
                  width: 12, 
                  height: 12, 
                  borderRadius: '50%', 
                  bgcolor: entry.color,
                  mr: 1 
                }} 
              />
              <Typography variant="body2" sx={{ color: theme.palette.text.primary }}>
                {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(3) : entry.value}
              </Typography>
            </Box>
          ))}
        </Card>
      );
    }
    return null;
  };

  const tabContent = [
    {
      label: 'Temporal Analysis',
      content: (
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                  Hourly Crowd Density & Risk Patterns
                </Typography>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data.timeData}>
                      <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
                      <XAxis 
                        dataKey="hour" 
                        stroke={theme.palette.text.secondary}
                        tick={{ fill: theme.palette.text.secondary }}
                      />
                      <YAxis 
                        yAxisId="left"
                        orientation="left"
                        stroke={theme.palette.info.main}
                        tick={{ fill: theme.palette.text.secondary }}
                      />
                      <YAxis 
                        yAxisId="right"
                        orientation="right"
                        stroke={theme.palette.error.main}
                        tick={{ fill: theme.palette.text.secondary }}
                      />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend />
                      <Bar 
                        yAxisId="left"
                        dataKey="crowd_density" 
                        name="Crowd Density"
                        fill={theme.palette.info.main} 
                        radius={[4, 4, 0, 0]}
                      />
                      <Bar 
                        yAxisId="right"
                        dataKey="risk_score" 
                        name="Risk Score"
                        fill={theme.palette.error.main} 
                        radius={[4, 4, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                  Peak Risk Hours
                </Typography>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={data.timeData
                          .sort((a, b) => b.risk_score - a.risk_score)
                          .slice(0, 6)
                          .map((d, i) => ({
                            ...d,
                            name: d.hour,
                            value: d.risk_score
                          }))
                        }
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {data.timeData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={theme.palette.error.light} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )
    },
    {
      label: 'Risk Distribution',
      content: (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                  Risk Level Distribution
                </Typography>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={data.riskDistribution}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {data.riskDistribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip content={<CustomTooltip />} />
                    </PieChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                  Feature Importance Analysis
                </Typography>
                <Box sx={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={data.featureImportance}>
                      <PolarGrid stroke={theme.palette.divider} />
                      <PolarAngleAxis dataKey="feature" stroke={theme.palette.text.secondary} />
                      <PolarRadiusAxis 
                        angle={30} 
                        domain={[0, 1]} 
                        stroke={theme.palette.text.secondary}
                      />
                      <Radar
                        name="Feature Weight"
                        dataKey="importance"
                        stroke={theme.palette.primary.main}
                        fill={theme.palette.primary.main}
                        fillOpacity={0.6}
                      />
                      <Tooltip />
                    </RadarChart>
                  </ResponsiveContainer>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )
    },
    {
      label: 'Spatial Analysis',
      content: (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
              Spatial Risk Distribution
            </Typography>
            <Box sx={{ height: 400 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                  margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                >
                  <CartesianGrid stroke={theme.palette.divider} strokeDasharray="3 3" />
                  <XAxis 
                    type="number" 
                    dataKey="x" 
                    name="X Position"
                    stroke={theme.palette.text.secondary}
                    tick={{ fill: theme.palette.text.secondary }}
                  />
                  <YAxis 
                    type="number" 
                    dataKey="y" 
                    name="Y Position"
                    stroke={theme.palette.text.secondary}
                    tick={{ fill: theme.palette.text.secondary }}
                  />
                  <ZAxis 
                    type="number" 
                    dataKey="risk" 
                    range={[100, 1000]} 
                    name="Risk Level"
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Scatter 
                    name="Risk Points" 
                    data={data.spatialData} 
                    fill={theme.palette.error.main}
                  >
                    {data.spatialData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.risk > 0.7 ? theme.palette.error.main :
                              entry.risk > 0.4 ? theme.palette.warning.main :
                              theme.palette.success.main}
                      />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </Box>
          </CardContent>
        </Card>
      )
    }
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card sx={{ bgcolor: 'background.paper' }}>
        <CardContent>
          <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
            Advanced Analytics Dashboard
          </Typography>
          
          <Tabs 
            value={activeTab} 
            onChange={handleTabChange}
            sx={{ mb: 3 }}
            variant="scrollable"
            scrollButtons="auto"
          >
            {tabContent.map((tab, index) => (
              <Tab 
                key={index}
                label={tab.label}
                sx={{ 
                  fontWeight: 500,
                  '&.Mui-selected': { color: 'primary.main' }
                }}
              />
            ))}
          </Tabs>
          
          <Box sx={{ minHeight: 400 }}>
            {tabContent[activeTab].content}
          </Box>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default AdvancedAnalytics;