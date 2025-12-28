import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroGrid}>
          <div className={styles.heroContent}>
            <Heading as="h1" className="hero__title">
              {siteConfig.title}
            </Heading>
            <p className="hero__subtitle">{siteConfig.tagline}</p>
            <div className={styles.buttons}>
              <Link
                className="button button--secondary button--lg"
                to="/docs/Module-1-Basics/Overview">
              Start Learning
              </Link>
            </div>
          </div>
          <div className={styles.heroImage}>
            <img
              src="/img/hero.png"
              alt="Physical AI & Humanoid Robotics Book Cover"
              className={styles.bookImage}
            />
          </div>
        </div>
      </div>
    </header>
  );
}

function ModulesSection() {
  const modules = [
    {
      id: 1,
      title: "The Robotic Nervous System (ROS 2)",
      description: "Introduction to Robot Operating System 2 (ROS 2), the middleware that serves as the nervous system for modern robotics applications.",
      path: "/docs/Module-1-Basics/Overview",
      color: "#2563eb"
    },
    {
      id: 2,
      title: "The Digital Twin (Gazebo & Unity)",
      description: "Focus on digital twin technology for humanoid robotics, specifically using Gazebo and Unity simulation environments.",
      path: "/docs/Module-2-Intermediate/Overview",
      color: "#7c3aed"
    },
    {
      id: 3,
      title: "The AI-Robot Brain (NVIDIA Isaac™)",
      description: "Implementation of AI capabilities in humanoid robots using NVIDIA's GPU-accelerated computing, perception algorithms, and control frameworks.",
      path: "/docs/Module-3-Advanced/Overview",
      color: "#f59e0b"
    },
    {
      id: 4,
      title: "Vision-Language-Action (VLA) Integration",
      description: "Integration of LLMs with humanoid robots for voice-command control and cognitive planning, covering multi-modal perception.",
      path: "/docs/Module-4-VLA/Overview",
      color: "#10b981"
    }
  ];

  return (
    <section className={styles.modulesSection}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2" className={styles.sectionTitle}>Learning Modules</Heading>
          <p className={styles.sectionSubtitle}>Explore our comprehensive curriculum designed to master humanoid robotics</p>
        </div>
        <div className={styles.modulesGrid}>
          {modules.map((module) => (
            <Link
              key={module.id}
              to={module.path}
              className={styles.moduleCard}
              style={{borderTopColor: module.color}}
            >
              <div className={styles.moduleNumber}>Module {module.id}</div>
              <Heading as="h3" className={styles.moduleTitle}>{module.title}</Heading>
              <p className={styles.moduleDescription}>{module.description}</p>
              <div className={styles.moduleLink}>
                Explore Module →
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Comprehensive documentation for Vision-Language-Action systems in robotics">
      <HomepageHeader />
      <main>
        <ModulesSection />
        <HomepageFeatures />
      </main>
    </Layout>
  );
}