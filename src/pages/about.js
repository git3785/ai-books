import React from 'react';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

export default function AboutPage() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title="About" description="About this documentation site">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--6 col--offset-3">
            <h1 className="hero__title">About This Documentation</h1>
            <p>{siteConfig.title} is a comprehensive guide for Vision-Language-Action systems in robotics.</p>
            <p>This documentation is built with Docusaurus to provide an accessible and well-organized resource for developers and researchers.</p>
          </div>
        </div>
      </div>
    </Layout>
  );
}