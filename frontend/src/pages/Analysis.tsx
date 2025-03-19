import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  Card,
  CardContent,
} from '@mui/material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import axios from 'axios';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analysis-tabpanel-${index}`}
      aria-labelledby={`analysis-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const Analysis: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisData, setAnalysisData] = useState<{
    shap_values: any[];
    lime_explanation: any[];
    feature_importance: any[];
  } | null>(null);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  useEffect(() => {
    const fetchAnalysisData = async () => {
      setLoading(true);
      setError(null);
      try {
        // Example patient data - in real app, this would come from the patient form
        const patientData = {
          clinical_data: {
            age: 65,
            gender: 'M',
            duration: 5,
            motor_UPDRS: 25,
            total_UPDRS: 35,
            tremor: 2,
            rigidity: 3,
            bradykinesia: 4,
          },
        };

        const response = await axios.post('http://localhost:8000/explain', patientData);
        setAnalysisData(response.data);
      } catch (err) {
        setError('Error fetching analysis data. Please try again.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalysisData();
  }, []);

  const renderShapChart = () => {
    if (!analysisData) return null;

    const data = {
      labels: analysisData.shap_values.map((item: any) => item.feature),
      datasets: [
        {
          label: 'SHAP Values',
          data: analysisData.shap_values.map((item: any) => item.importance),
          backgroundColor: 'rgba(54, 162, 235, 0.5)',
        },
      ],
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          position: 'top' as const,
        },
        title: {
          display: true,
          text: 'Feature Importance (SHAP)',
        },
      },
    };

    return <Bar data={data} options={options} />;
  };

  const renderLimeChart = () => {
    if (!analysisData) return null;

    const data = {
      labels: analysisData.lime_explanation.map((item: any) => item.feature),
      datasets: [
        {
          label: 'LIME Values',
          data: analysisData.lime_explanation.map((item: any) => item.importance),
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
        },
      ],
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          position: 'top' as const,
        },
        title: {
          display: true,
          text: 'Feature Importance (LIME)',
        },
      },
    };

    return <Bar data={data} options={options} />;
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Model Analysis
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Paper sx={{ width: '100%' }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          aria-label="analysis tabs"
        >
          <Tab label="SHAP Analysis" />
          <Tab label="LIME Analysis" />
          <Tab label="Feature Importance" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <Card>
            <CardContent>
              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              ) : (
                renderShapChart()
              )}
            </CardContent>
          </Card>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Card>
            <CardContent>
              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              ) : (
                renderLimeChart()
              )}
            </CardContent>
          </Card>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            {analysisData?.feature_importance.map((item: any, index: number) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {item.feature}
                    </Typography>
                    <Typography color="textSecondary">
                      Importance: {item.importance.toFixed(4)}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default Analysis; 