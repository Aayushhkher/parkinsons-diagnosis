import React from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import PersonAddIcon from '@mui/icons-material/PersonAdd';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import AssessmentIcon from '@mui/icons-material/Assessment';
import TimelineIcon from '@mui/icons-material/Timeline';
import WarningIcon from '@mui/icons-material/Warning';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const Dashboard: React.FC = () => {
  const navigate = useNavigate();

  const quickActions = [
    {
      title: 'New Patient Assessment',
      description: 'Start a new patient evaluation',
      icon: <PersonAddIcon />,
      path: '/patient-form',
    },
    {
      title: 'Batch Analysis',
      description: 'Upload and analyze multiple patient records',
      icon: <UploadFileIcon />,
      path: '/batch-upload',
    },
    {
      title: 'Model Analysis',
      description: 'View model performance and explanations',
      icon: <AnalyticsIcon />,
      path: '/analysis',
    },
    {
      title: 'Reports',
      description: 'Generate and view analysis reports',
      icon: <AssessmentIcon />,
      path: '/reports',
    },
  ];

  const systemStatus = [
    {
      title: 'Model Status',
      status: 'Active',
      icon: <CheckCircleIcon color="success" />,
    },
    {
      title: 'API Status',
      status: 'Connected',
      icon: <CheckCircleIcon color="success" />,
    },
    {
      title: 'Database Status',
      status: 'Connected',
      icon: <CheckCircleIcon color="success" />,
    },
  ];

  const recentActivity = [
    {
      title: 'New Patient Assessment',
      description: 'Patient ID: 12345',
      timestamp: '2 minutes ago',
      icon: <PersonAddIcon />,
    },
    {
      title: 'Batch Analysis Complete',
      description: 'Processed 50 records',
      timestamp: '1 hour ago',
      icon: <UploadFileIcon />,
    },
    {
      title: 'Model Update',
      description: 'New model version deployed',
      timestamp: '2 hours ago',
      icon: <TimelineIcon />,
    },
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* Quick Actions */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Quick Actions
            </Typography>
            <Grid container spacing={2}>
              {quickActions.map((action) => (
                <Grid item xs={12} sm={6} key={action.title}>
                  <Card>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        {action.icon}
                        <Typography variant="h6" sx={{ ml: 1 }}>
                          {action.title}
                        </Typography>
                      </Box>
                      <Typography color="textSecondary">
                        {action.description}
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <Button
                        size="small"
                        onClick={() => navigate(action.path)}
                      >
                        Go to
                      </Button>
                    </CardActions>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>

        {/* System Status */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              System Status
            </Typography>
            <List>
              {systemStatus.map((status) => (
                <ListItem key={status.title}>
                  <ListItemIcon>
                    {status.icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={status.title}
                    secondary={status.status}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Activity
            </Typography>
            <List>
              {recentActivity.map((activity) => (
                <ListItem key={activity.title}>
                  <ListItemIcon>
                    {activity.icon}
                  </ListItemIcon>
                  <ListItemText
                    primary={activity.title}
                    secondary={
                      <React.Fragment>
                        <Typography
                          component="span"
                          variant="body2"
                          color="textPrimary"
                        >
                          {activity.description}
                        </Typography>
                        {' — '}
                        {activity.timestamp}
                      </React.Fragment>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard; 