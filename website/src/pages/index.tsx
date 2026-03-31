import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';


const codeSnippet = `from ml_dash import Experiment

with Experiment(
    prefix="my-user/my-project/run-001",
    dash_url="https://api.dash.ml",
).run as exp:

    # Log hyperparameters
    exp.params.set(lr=0.001, batch_size=32, arch="resnet50")

    for epoch in range(100):
        loss, acc = train_epoch()

        # Non-blocking — never slows your loop
        exp.metrics("train").log(
            loss=loss, accuracy=acc, epoch=epoch
        )

    # Upload final checkpoint
    exp.files("checkpoints").save("model.pth")`;

export default function Home() {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout description={siteConfig.tagline}>
      <main className={styles.main}>

        {/* Hero */}
        <section className={styles.hero}>
          <div className={styles.heroContent}>
            <h1 className={styles.heroTitle}>
              Experiment tracking<br />that gets out of your way
            </h1>
            <p className={styles.heroSub}>
              Log metrics, parameters, and files from your training loop.
              Non-blocking, resilient, and ready for distributed sweeps.
            </p>

            <div className={styles.heroInstall}>
              <span className={styles.installPrompt}>$</span>
              <code>pip install ml-dash</code>
            </div>

            <div className={styles.heroActions}>
              <Link className={`button button--primary button--lg ${styles.btnPrimary}`} to="/docs/getting-started">
                Get Started →
              </Link>
              <Link className={`button button--secondary button--lg ${styles.btnSecondary}`} to="/docs">
                Documentation
              </Link>
            </div>
          </div>
        </section>

        <hr className={styles.divider} />

        {/* Code + pitch */}
        <section className={styles.showcase}>
          <div className={styles.showcasePitch}>
            <h2>Everything you need. Nothing you don't.</h2>
            <p>
              A minimal surface area covering the full ML experiment lifecycle —
              from hyperparameter logging to checkpoint upload —
              without config files or a running daemon.
            </p>
            <ul className={styles.pitchList}>
              <li>Fluent, chainable API</li>
              <li>Automatic background batching</li>
              <li>Local-first, remote-optional</li>
              <li>Retry + re-queue on network failure</li>
            </ul>
          </div>
          <div className={styles.showcaseCode}>
            <div className={styles.codeWindow}>
              <div className={styles.codeWindowBar}>
                <span /><span /><span />
                <span className={styles.codeWindowTitle}>train.py</span>
              </div>
              <pre className={styles.code}>{codeSnippet}</pre>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className={styles.cta}>
          <h2>Ready to start?</h2>
          <p>Set up in under 5 minutes. No account required for local mode.</p>
          <Link className={`button button--primary button--lg ${styles.btnPrimary}`} to="/docs/getting-started">
            Read the docs →
          </Link>
        </section>

      </main>
    </Layout>
  );
}
