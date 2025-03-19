import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  CircularProgress,
  Alert,
  Stepper,
  Step,
  StepLabel,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

interface PatientData {
  clinical_data: {
    age: number;
    gender: string;
    duration: number;
    motor_UPDRS: number;
    total_UPDRS: number;
    tremor: number;
    rigidity: number;
    bradykinesia: number;
  };
  imaging_data?: {
    mri_features: number[];
    pet_features: number[];
  };
  voice_data?: {
    jitter: number;
    shimmer: number;
    nhr: number;
    hnr: number;
    rpde: number;
    dfa: number;
    spread1: number;
    spread2: number;
    ppe: number;
  };
  motion_data?: {
    acceleration: number[];
    gyroscope: number[];
    magnetometer: number[];
  };
}

const steps = ['Clinical Data', 'Imaging Data', 'Voice Data', 'Motion Data'];

const PatientForm: React.FC = () => {
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<{
    prediction: number;
    probability: number;
  } | null>(null);
  const [formData, setFormData] = useState<PatientData>({
    clinical_data: {
      age: 0,
      gender: '',
      duration: 0,
      motor_UPDRS: 0,
      total_UPDRS: 0,
      tremor: 0,
      rigidity: 0,
      bradykinesia: 0,
    },
  });

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:8000/predict', formData);
      setPrediction(response.data);
    } catch (err) {
      setError('Error making prediction. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field: string, value: any) => {
    setFormData((prev) => ({
      ...prev,
      clinical_data: {
        ...prev.clinical_data,
        [field]: value,
      },
    }));
  };

  const renderClinicalDataStep = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          label="Age"
          type="number"
          value={formData.clinical_data.age}
          onChange={(e) => handleInputChange('age', Number(e.target.value))}
        />
      </Grid>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          label="Gender"
          value={formData.clinical_data.gender}
          onChange={(e) => handleInputChange('gender', e.target.value)}
        />
      </Grid>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          label="Disease Duration (years)"
          type="number"
          value={formData.clinical_data.duration}
          onChange={(e) => handleInputChange('duration', Number(e.target.value))}
        />
      </Grid>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          label="Motor UPDRS"
          type="number"
          value={formData.clinical_data.motor_UPDRS}
          onChange={(e) => handleInputChange('motor_UPDRS', Number(e.target.value))}
        />
      </Grid>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          label="Total UPDRS"
          type="number"
          value={formData.clinical_data.total_UPDRS}
          onChange={(e) => handleInputChange('total_UPDRS', Number(e.target.value))}
        />
      </Grid>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          label="Tremor Score"
          type="number"
          value={formData.clinical_data.tremor}
          onChange={(e) => handleInputChange('tremor', Number(e.target.value))}
        />
      </Grid>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          label="Rigidity Score"
          type="number"
          value={formData.clinical_data.rigidity}
          onChange={(e) => handleInputChange('rigidity', Number(e.target.value))}
        />
      </Grid>
      <Grid item xs={12} sm={6}>
        <TextField
          fullWidth
          label="Bradykinesia Score"
          type="number"
          value={formData.clinical_data.bradykinesia}
          onChange={(e) => handleInputChange('bradykinesia', Number(e.target.value))}
        />
      </Grid>
    </Grid>
  );

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        New Patient Assessment
      </Typography>
      
      <Paper sx={{ p: 3, mb: 3 }}>
        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {prediction && (
          <Alert severity={prediction.prediction === 1 ? 'warning' : 'success'} sx={{ mb: 2 }}>
            Prediction: {prediction.prediction === 1 ? 'PD Positive' : 'PD Negative'}
            <br />
            Probability: {(prediction.probability * 100).toFixed(2)}%
          </Alert>
        )}

        {renderClinicalDataStep()}

        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 3 }}>
          <Button
            disabled={activeStep === 0}
            onClick={handleBack}
            sx={{ mr: 1 }}
          >
            Back
          </Button>
          {activeStep === steps.length - 1 ? (
            <Button
              variant="contained"
              onClick={handleSubmit}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} /> : 'Submit'}
            </Button>
          ) : (
            <Button
              variant="contained"
              onClick={handleNext}
            >
              Next
            </Button>
          )}
        </Box>
      </Paper>
    </Box>
  );
};

export default PatientForm; 