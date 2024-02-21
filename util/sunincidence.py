import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from util.sunposition import sunpos

############
# REFERENCES
# Solar Position Code...https://github.com/s-bear/sun-position
# Solar Position Concepts...http://www.me.umn.edu/courses/me4131/LabManual/AppDSolarRadiation.pdf
# Data Validation...https://www.nrel.gov/midc/solpos/solpos.html

# FUNCTION: convert degrees to radians
def deg2rad(deg):
    rad = deg / 180.0 * math.pi
    return rad
    
# FUNCTION: convert radians to degrees
def rad2deg(rad):
    deg = rad / math.pi * 180.0
    return deg
    
# FUNCTION: get solar position data
def getSolarPosition(t, project_data, return_datum=False):

    # Get solar position from lat/lng, elevation and datetime
    phi, theta_h, rasc, d, h = sunpos(t, project_data['latitude'], project_data['longitude'], project_data['elevation'])[:5]
    
    # Calculate tilt angle from vertical
    eta = 90 - project_data['tilt']
    
    # Calculate surface-solar azimuth angle
    gamma = math.fabs((phi - project_data['azimuth']))
    
    if project_data['zenith_filter'] and theta_h > project_data['zenith_limit']:
        theta_h = project_data['zenith_limit']

    # Calculate altitude angle
    beta = 90.0 - theta_h
    
    # Calculate incident angle to surface
    incident_rad = math.acos((math.cos(deg2rad(beta)) * math.cos(deg2rad(gamma)) * math.sin(deg2rad(eta))) + (math.sin(deg2rad(beta)) * math.cos(deg2rad(eta))))
    
    # Solar position datum
    zenith_rad = np.deg2rad(theta_h)
    if return_datum:
        return {
            'Datetime_UTC': t,
            'Azimuth': phi,
            'Zenith': zenith_rad,
            'RightAscension': rasc,
            'Declination': d,
            'HourAngle': h,
            'IncidentAngle': incident_rad 
        }
    return zenith_rad, incident_rad
    
    # return sp_datum


def siteinfo2projectdata(lat, long, orientation, tilt, interval=5):
    return {
        'latitude': lat,
        'longitude': long,
        'elevation': 0,
        'tilt': tilt,
        'azimuth': orientation,
        'zenith_limit': 90,
        'zenith_filter': False,
        'interval': interval
    }


def siteinfo2incidence(timestamp, lat, long, orientation, tilt, return_datum=False):
    return getSolarPosition(timestamp, siteinfo2incidence(lat, long, orientation, tilt), return_datum=return_datum)['IncidentAngle']
        
# FUNCTION: loop through timestamp array, calculate solar position
def loopSolarPositionByProject(start: datetime.datetime, end: datetime.datetime, project_data, return_datum=False):

    # Solar position data array
    sp_data = []
    
    # Set start timestamp
    dt = start
    
    # Set timestamp invertal
    delta = datetime.timedelta(minutes=project_data['interval'])

    # Loop through timestamps...
    while dt <= end:

        # Print timestamp
        #print dt.strftime("%Y-%m-%dT%H:%M")
        
        sp_datum = getSolarPosition(dt, project_data, return_datum=return_datum)
        
        # Add solar position datum to data array
        sp_data.append(sp_datum)
        
        # Increment timestamp by +1 delta
        dt += delta
        
    return sp_data


def get_day_plot(day: datetime.date, project_data):
    """
    Example project data object

    project_data = {
        'latitude': p_latitude_dd,
        'longitude': p_longitude_dd,
        'elevation': p_elevation_m,
        'tilt': p_tilt_deg,
        'azimuth': p_azimuth_deg,
        'zenith_limit': p_zenith_limit,
        'zenith_filter': p_zenith_filter,
        'interval': p_interval_minutes
    }
    """
    print(project_data)
    print('Looping through solar position calcs...')
    start = datetime.datetime.combine(day, datetime.time(0, 0))
    end = start + datetime.timedelta(days=1)
    sp_data = loopSolarPositionByProject(start, end, project_data, return_datum=True)
    print('Done!')

    o_datetime_utc = [x['Datetime_UTC'] for x in sp_data]        
    o_azimuth = [x['Azimuth'] for x in sp_data]        
    o_zenith = [np.rad2deg(x['Zenith']) for x in sp_data]
    o_incident_angle = [np.rad2deg(x['IncidentAngle']) for x in sp_data]

    title = 'Solar Incident Angle @ (' + str(project_data['latitude']) + ',' + str(project_data['longitude']) + ')'
    title += '\nTilt (Horizontal): ' + str(project_data['tilt']) + ' deg, Azimuth (South CC): ' + str(project_data['azimuth']) + ' deg @ Elevation: ' + str(project_data['elevation']) + ' m'
    title += '\n' + str(start) + ' to ' + str(end) + ' @ ' + str(project_data['interval']) + ' min Interval'

    f, ax = plt.subplots(figsize=(16,8))
    ax.plot(o_datetime_utc, o_azimuth, 'b', label='Azimuth (South CC) [deg]')
    ax.plot(o_datetime_utc, o_zenith, 'g', label='Zenith (Vertical) [deg]')
    ax.plot(o_datetime_utc, o_incident_angle, 'r', label='Incident Angle [deg]')
    ax.set_title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    # place xticks every hour
    plt.xticks(o_datetime_utc[::6], rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    plt.yticks(np.arange(0, 360, 15))
    plt.grid()
    plt.show()


def day_plot_with_pv(day: datetime.date, project_data, pv_data):
    """
    Example project data object

    project_data = {
        'latitude': p_latitude_dd,
        'longitude': p_longitude_dd,
        'elevation': p_elevation_m,
        'tilt': p_tilt_deg,
        'azimuth': p_azimuth_deg,
        'zenith_limit': p_zenith_limit,
        'zenith_filter': p_zenith_filter,
        'interval': p_interval_minutes
    }
    """
    print(project_data)
    print('Looping through solar position calcs...')
    start = datetime.datetime.combine(day, datetime.time(0, 0))
    end = start + datetime.timedelta(days=1)
    sp_data = loopSolarPositionByProject(start, end, project_data, return_datum=True)
    print('Done!')

    o_datetime_utc = [x['Datetime_UTC'] for x in sp_data]
    o_zenith = [np.rad2deg(x['Zenith']) for x in sp_data]
    o_incident_angle = [np.rad2deg(x['IncidentAngle']) for x in sp_data]

    title = 'Solar Incident Angle @ (' + str(project_data['latitude']) + ',' + str(project_data['longitude']) + ')'
    title += '\nTilt (Horizontal): ' + str(project_data['tilt']) + ' deg, Azimuth (South CC): ' + str(project_data['azimuth']) + ' deg @ Elevation: ' + str(project_data['elevation']) + ' m'
    title += '\n' + str(start) + ' to ' + str(end) + ' @ ' + str(project_data['interval']) + ' min Interval'

    f, ax = plt.subplots(figsize=(16,6))
    ax.plot(o_datetime_utc, o_zenith, 'g', label='Zenith (Vertical) [deg]')
    ax.plot(o_datetime_utc, o_incident_angle, 'r', label='Incident Angle [deg]')
    ax.axhline(y=90, color='k', linestyle='--', label='90 deg')
    ax.invert_yaxis()
    ax.set_title(title)
    plt.yticks(np.arange(0, 181, 15))

    plt.legend(loc='upper right')
    plt.grid()

    ax2 = ax.twinx()
    ax2.plot(o_datetime_utc[:len(pv_data)], pv_data, 'b', label='PV Power [W]')
    ax2.set_ylabel('PV Power [W]')
    ax2.set_ylim(0, 1)

    plt.xticks(o_datetime_utc[::6], rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    plt.legend(loc='upper left')
    f.tight_layout()
    plt.show()