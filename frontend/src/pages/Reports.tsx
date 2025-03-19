import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Button,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import DownloadIcon from '@mui/icons-material/Download';
import PrintIcon from '@mui/icons-material/Print';
import ShareIcon from '@mui/icons-material/Share';

interface ReportData {
  total_patients: number;
  pd_positive: number;
  pd_negative: number;
  average_age: number;
  gender_distribution: {
    male: number;
    female: number;
  };
  model_performance: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
  recent_predictions: Array<{
    id: string;
    date: string;
    prediction: number;
    probability: number;
  }>;
}

const Reports: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [reportType, setReportType] = useState('daily');
  const [startDate, setStartDate] = useState<Date | null>(new Date());
  const [endDate, setEndDate] = useState<Date | null>(new Date());

  const handleGenerateReport = async () => {
    setLoading(true);
    setError(null);
    try {
      // In a real application, this would call the backend API
      // For now, we'll use mock data
      const mockData: ReportData = {
        total_patients: 150,
        pd_positive: 45,
        pd_negative: 105,
        average_age: 65.5,
        gender_distribution: {
          male: 85,
          female: 65,
        },
        model_performance: {
          accuracy: 0.92,
          precision: 0.89,
          recall: 0.91,
          f1_score: 0.90,
        },
        recent_predictions: [
          {
            id: 'P001',
            date: '2024-03-20',
            prediction: 1,
            probability: 0.85,
          },
          {
            id: 'P002',
            date: '2024-03-20',
            prediction: 0,
            probability: 0.92,
          },
        ],
      };
      setReportData(mockData);
    } catch (err) {
      setError('Error generating report. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = (format: 'pdf' | 'csv' | 'excel') => {
    // Implement export functionality
    console.log(`Exporting report in ${format} format`);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Reports
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} sm={4}>
            <FormControl fullWidth>
              <InputLabel>Report Type</InputLabel>
              <Select
                value={reportType}
                label="Report Type"
                onChange={(e) => setReportType(e.target.value)}
              >
                <MenuItem value="daily">Daily Report</MenuItem>
                <MenuItem value="weekly">Weekly Report</MenuItem>
                <MenuItem value="monthly">Monthly Report</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={4}>
            <LocalizationProvider dateAdapter={AdapterDateFns}>
              <DatePicker
                label="Start Date"
                value={startDate}
                onChange={(newValue) => setStartDate(newValue)}
                sx={{ width: '100%' }}
              />
            </LocalizationProvider>
          </Grid>
          <Grid item xs={12} sm={4}>
            <LocalizationProvider dateAdapter={AdapterDateFns}>
              <DatePicker
                label="End Date"
                value={endDate}
                onChange={(newValue) => setEndDate(newValue)}
                sx={{ width: '100%' }}
              />
            </LocalizationProvider>
          </Grid>
          <Grid item xs={12}>
            <Button
              variant="contained"
              onClick={handleGenerateReport}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} /> : 'Generate Report'}
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {reportData && (
        <>
          <Grid container spacing={3}>
            {/* Summary Cards */}
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Total Patients
                  </Typography>
                  <Typography variant="h4">
                    {reportData.total_patients}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    PD Positive
                  </Typography>
                  <Typography variant="h4" color="error">
                    {reportData.pd_positive}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    PD Negative
                  </Typography>
                  <Typography variant="h4" color="success.main">
                    {reportData.pd_negative}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={3}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Average Age
                  </Typography>
                  <Typography variant="h4">
                    {reportData.average_age}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            {/* Model Performance */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Model Performance
                </Typography>
                <TableContainer>
                  <Table>
                    <TableBody>
                      <TableRow>
                        <TableCell>Accuracy</TableCell>
                        <TableCell>{(reportData.model_performance.accuracy * 100).toFixed(2)}%</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Precision</TableCell>
                        <TableCell>{(reportData.model_performance.precision * 100).toFixed(2)}%</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Recall</TableCell>
                        <TableCell>{(reportData.model_performance.recall * 100).toFixed(2)}%</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>F1 Score</TableCell>
                        <TableCell>{(reportData.model_performance.f1_score * 100).toFixed(2)}%</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Grid>

            {/* Gender Distribution */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Gender Distribution
                </Typography>
                <TableContainer>
                  <Table>
                    <TableBody>
                      <TableRow>
                        <TableCell>Male</TableCell>
                        <TableCell>{reportData.gender_distribution.male}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Female</TableCell>
                        <TableCell>{reportData.gender_distribution.female}</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Grid>

            {/* Recent Predictions */}
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="h6">
                    Recent Predictions
                  </Typography>
                  <Box>
                    <Button
                      startIcon={<DownloadIcon />}
                      onClick={() => handleExport('pdf')}
                      sx={{ mr: 1 }}
                    >
                      PDF
                    </Button>
                    <Button
                      startIcon={<DownloadIcon />}
                      onClick={() => handleExport('csv')}
                      sx={{ mr: 1 }}
                    >
                      CSV
                    </Button>
                    <Button
                      startIcon={<PrintIcon />}
                      onClick={() => window.print()}
                      sx={{ mr: 1 }}
                    >
                      Print
                    </Button>
                    <Button
                      startIcon={<ShareIcon />}
                      onClick={() => handleExport('excel')}
                    >
                      Share
                    </Button>
                  </Box>
                </Box>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Patient ID</TableCell>
                        <TableCell>Date</TableCell>
                        <TableCell>Prediction</TableCell>
                        <TableCell>Probability</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {reportData.recent_predictions.map((prediction) => (
                        <TableRow key={prediction.id}>
                          <TableCell>{prediction.id}</TableCell>
                          <TableCell>{prediction.date}</TableCell>
                          <TableCell>
                            {prediction.prediction === 1 ? 'PD Positive' : 'PD Negative'}
                          </TableCell>
                          <TableCell>
                            {(prediction.probability * 100).toFixed(2)}%
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Grid>
          </Grid>
        </>
      )}
    </Box>
  );
};

export default Reports; 