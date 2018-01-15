/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	num_particles = 100;
	
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++)
	{
		Particle part;
		part.x = dist_x(gen);
		part.y = dist_y(gen);
		part.theta = dist_theta(gen);
		part.weight = 1.0;
		particles.push_back(part);
		weights.push_back(1.0);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (int i = 0; i < particles.size(); i++)
	{
		double theta = particles[i].theta;
		if (fabs(yaw_rate) > 0.001)
		{
			particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
			particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
		else
		{
			particles[i].x += velocity * delta_t * cos(theta);
			particles[i].y += velocity * delta_t * sin(theta);
		}
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
	for (int i = 0; i < observations.size(); i++)
	{
		LandmarkObs obs = observations[i];
		double minimum_distance = dist(predicted[0].x, predicted[0].y, obs.x, obs.y);

		for (auto pred: predicted)
		{
			double observed_distance = dist(pred.x, pred.y, obs.x, obs.y);

			if (observed_distance <= minimum_distance)
			{
				minimum_distance = observed_distance;
				observations[i].id = pred.id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	double sigx = std_landmark[0];
	double sigy = std_landmark[1];
	double gauss_const = (1. / (2. * M_PI * sigx * sigy));
	double sigx2 = 2 * pow(sigx, 2);
	double sigy2 = 2 * pow(sigy, 2);

	for (int i = 0; i < num_particles; i++)
	{
		Particle particle = particles[i];

		std::vector<LandmarkObs> map_obs = std::vector<LandmarkObs>();
		for (int j = 0; j < observations.size(); j++)
		{
			LandmarkObs map_obs_trans;
			map_obs_trans.id = observations[j].id;
			map_obs_trans.x = particle.x + (cos(particle.theta) * observations[j].x) - (sin(particle.theta) * observations[j].y);
			map_obs_trans.y = particle.y + (sin(particle.theta) * observations[j].x) + (cos(particle.theta) * observations[j].y);
			map_obs.push_back(map_obs_trans);
		}
		
		std::vector<LandmarkObs> predicted;
		for (auto landmark: map_landmarks.landmark_list)
		{
			double distance = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);
			if (distance <= sensor_range)
			{
				LandmarkObs map_coordinate_observation;
				map_coordinate_observation.id = landmark.id_i;
				map_coordinate_observation.x = landmark.x_f;
				map_coordinate_observation.y = landmark.y_f;
				predicted.push_back(map_coordinate_observation);
			}
		}

		dataAssociation(predicted, map_obs);

		double proba = 1;
		for (auto obs: map_obs)
		{
			double x = obs.x;
			double y = obs.y;
			LandmarkObs pred;

			for (int k = 0; k < predicted.size(); k++)
			{
				if (predicted[k].id == obs.id)
				{
					pred = predicted[k];
					break;
				}
			}
			double mu_x = pred.x;
			double mu_y = pred.y;
			proba *= gauss_const * exp(-(pow(x - mu_x, 2) / sigx2 + (pow(y - mu_y, 2) / sigy2)));
		}
		weights[i] = proba;
		particles[i].weight = proba;
	}
}



void ParticleFilter::resample()
{
	std::default_random_engine random_engine;
	std::discrete_distribution<> distribution(weights.begin(), weights.end());
	std::vector<Particle> resampled_particles = std::vector<Particle>();
	while (resampled_particles.size() < num_particles)
	{
		int i = distribution(random_engine);
		resampled_particles.push_back(particles[i]);
	}
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
