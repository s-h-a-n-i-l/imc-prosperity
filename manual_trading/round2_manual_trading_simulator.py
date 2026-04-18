from __future__ import annotations

import json
import uuid
from typing import Any

from IPython.display import HTML, display


TOTAL_BUDGET = 50_000
SIMULATED_PLAYERS = 19999
TOTAL_PLAYERS = SIMULATED_PLAYERS + 1
MIN_SPEED_INVESTMENT = 0
MAX_SPEED_INVESTMENT = 88
INITIAL_ERROR_STD = 5
ERROR_STD_DECAY = 0.1
FROZEN_PLAYERS_MEAN = 10.0
FROZEN_PLAYERS_VARIANCE = 3.0


def build_round2_manual_trading_simulator_html(container_id: str | None = None) -> str:
    """Return the notebook HTML for the interactive round 2 manual-trading simulator."""
    container_id = container_id or f"round2-manual-sim-{uuid.uuid4().hex}"
    config: dict[str, Any] = {
        "budget": TOTAL_BUDGET,
        "simulatedPlayers": SIMULATED_PLAYERS,
        "totalPlayers": TOTAL_PLAYERS,
        "minInvestment": MIN_SPEED_INVESTMENT,
        "maxInvestment": MAX_SPEED_INVESTMENT,
        "initialErrorStd": INITIAL_ERROR_STD,
        "errorStdDecay": ERROR_STD_DECAY,
        "frozenPlayersMean": FROZEN_PLAYERS_MEAN,
        "frozenPlayersVariance": FROZEN_PLAYERS_VARIANCE,
    }

    template = r"""
<div id=__CONTAINER_ID__></div>
<script type="text/javascript">
(function () {
  const containerId = __CONTAINER_ID__;
  const config = __CONFIG_JSON__;
  const root = document.getElementById(containerId);

  if (!root) {
    return;
  }

  const registry = window.manualTradingRound2SimulatorRegistry = window.manualTradingRound2SimulatorRegistry || {};
  const researchValues = Array.from({ length: 101 }, (_, x) => (
    x <= 0 ? 0 : 200000 * Math.log(1 + x) / Math.log(101)
  ));
  const scaleUnit = 7 / 100;
  const scenarioCache = new Map();

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  function formatInt(value) {
    return Math.round(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
  }

  function formatPct(value) {
    return `${value}%`;
  }

  function randomInteger(minimum, maximum) {
    return Math.floor(Math.random() * (maximum - minimum + 1)) + minimum;
  }

  function randomChoice(values) {
    return values[Math.floor(Math.random() * values.length)];
  }

  function gaussian(mean, std) {
    if (std <= 0) {
      return mean;
    }

    let u = 0;
    let v = 0;

    while (u === 0) {
      u = Math.random();
    }

    while (v === 0) {
      v = Math.random();
    }

    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return mean + std * z;
  }

  function laplace(location, scale) {
    if (scale <= 0) {
      return location;
    }

    const u = Math.random() - 0.5;
    const sign = u < 0 ? -1 : 1;
    return location - scale * sign * Math.log(1 - 2 * Math.abs(u));
  }

  function rankToMultiplier(rank, totalPlayers) {
    if (totalPlayers <= 1) {
      return 0.9;
    }

    return 0.9 - (0.8 * (rank - 1) / (totalPlayers - 1));
  }

  function buildCounts(investments) {
    const counts = new Array(config.maxInvestment + 1).fill(0);
    for (const investment of investments) {
      counts[investment] += 1;
    }
    return counts;
  }

  function buildGreaterCounts(counts) {
    const greaterCounts = new Array(counts.length).fill(0);
    let running = 0;

    for (let speed = counts.length - 1; speed >= 0; speed -= 1) {
      greaterCounts[speed] = running;
      running += counts[speed];
    }

    return greaterCounts;
  }

  function computeBestScenario(speedPct, speedMultiplier) {
    const cacheKey = `${speedPct}|${speedMultiplier.toFixed(12)}`;
    const cached = scenarioCache.get(cacheKey);

    if (cached) {
      return cached;
    }

    const cap = 100 - speedPct;
    let best = {
      pnl: Number.NEGATIVE_INFINITY,
      gross: 0,
      researchPct: 0,
      scalePct: 0,
      used: config.budget * speedPct / 100,
    };

    for (let researchPct = 0; researchPct <= cap; researchPct += 1) {
      const researchValue = researchValues[researchPct];
      const maxScalePct = cap - researchPct;
      const grossPerScalePct = researchValue * scaleUnit * speedMultiplier;
      const scalePct = grossPerScalePct > config.budget / 100 ? maxScalePct : 0;
      const gross = researchValue * (scalePct * scaleUnit) * speedMultiplier;
      const used = config.budget * (researchPct + scalePct + speedPct) / 100;
      const pnl = gross - used;

      if (pnl > best.pnl + 1e-9) {
        best = {
          pnl,
          gross,
          researchPct,
          scalePct,
          used,
        };
      }
    }

    scenarioCache.set(cacheKey, best);
    return best;
  }

  function computePlayerOutcome(speedPct, greaterCounts) {
    const rank = 1 + greaterCounts[speedPct];
    const speedMultiplier = rankToMultiplier(rank, config.totalPlayers);
    const scenario = computeBestScenario(speedPct, speedMultiplier);

    return {
      speedPct,
      rank,
      speedMultiplier,
      ...scenario,
    };
  }

  function computeCurve(simulatedInvestments) {
    const counts = buildCounts(simulatedInvestments);
    const greaterCounts = buildGreaterCounts(counts);
    const curve = [];

    for (let speedPct = config.minInvestment; speedPct <= config.maxInvestment; speedPct += 1) {
      curve.push(computePlayerOutcome(speedPct, greaterCounts));
    }

    return curve;
  }

  function buildLookup(simulatedInvestments, userInvestment) {
    const counts = buildCounts([...simulatedInvestments, userInvestment]);
    const greaterCounts = buildGreaterCounts(counts);
    const lookup = {};

    for (let currentPct = config.minInvestment; currentPct <= config.maxInvestment; currentPct += 1) {
      let bestPnl = Number.NEGATIVE_INFINITY;
      let bestTargets = [];

      for (let targetPct = config.minInvestment; targetPct <= config.maxInvestment; targetPct += 1) {
        const adjustedGreater = greaterCounts[targetPct] - (currentPct > targetPct ? 1 : 0);
        const rank = 1 + adjustedGreater;
        const speedMultiplier = rankToMultiplier(rank, config.totalPlayers);
        const scenario = computeBestScenario(targetPct, speedMultiplier);

        if (scenario.pnl > bestPnl + 1e-9) {
          bestPnl = scenario.pnl;
          bestTargets = [targetPct];
        } else if (Math.abs(scenario.pnl - bestPnl) <= 1e-9) {
          bestTargets.push(targetPct);
        }
      }

      lookup[currentPct] = {
        bestPnl,
        bestTargets,
      };
    }

    return lookup;
  }

  function computeTopCurvePoints(curve, limit) {
    return [...curve]
      .sort((left, right) => right.pnl - left.pnl || left.speedPct - right.speedPct)
      .slice(0, limit);
  }

  function computeMostCommonSpeeds(investments, userInvestment, limit) {
    const counts = buildCounts([...investments, userInvestment]);

    return counts
      .map((count, speedPct) => ({ speedPct, count }))
      .filter((point) => point.count > 0)
      .sort((left, right) => right.count - left.count || left.speedPct - right.speedPct)
      .slice(0, limit);
  }

  function sampleFrozenCount() {
    const std = Math.sqrt(config.frozenPlayersVariance);
    return clamp(
      Math.round(gaussian(config.frozenPlayersMean, std)),
      0,
      config.simulatedPlayers,
    );
  }

  function randomIndexSet(totalCount, chosenCount) {
    const selected = new Set();

    while (selected.size < chosenCount) {
      selected.add(randomInteger(0, totalCount - 1));
    }

    return selected;
  }

  function buildLineChart(curve, currentUserInvestment) {
    const width = 980;
    const height = 420;
    const margin = { top: 30, right: 28, bottom: 56, left: 86 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const pnlValues = curve.map((point) => point.pnl);
    const minPnl = Math.min(...pnlValues);
    const maxPnl = Math.max(...pnlValues);
    const yPadding = Math.max(1_000, (maxPnl - minPnl) * 0.08);
    const yMin = minPnl - yPadding;
    const yMax = maxPnl + yPadding;

    function xFor(speedPct) {
      const span = config.maxInvestment - config.minInvestment || 1;
      return margin.left + innerWidth * (speedPct - config.minInvestment) / span;
    }

    function yFor(pnl) {
      const span = yMax - yMin || 1;
      return margin.top + innerHeight * (1 - (pnl - yMin) / span);
    }

    const points = curve.map((point) => `${xFor(point.speedPct).toFixed(2)},${yFor(point.pnl).toFixed(2)}`).join(" ");
    const currentPoint = curve.find((point) => point.speedPct === currentUserInvestment) || curve[0];
    const bestPoint = computeTopCurvePoints(curve, 1)[0];

    const xTicks = [];
    for (let tick = config.minInvestment; tick <= config.maxInvestment; tick += 11) {
      xTicks.push(tick);
    }
    if (xTicks[xTicks.length - 1] !== config.maxInvestment) {
      xTicks.push(config.maxInvestment);
    }

    const yTicks = [];
    for (let index = 0; index < 5; index += 1) {
      yTicks.push(yMin + (index * (yMax - yMin) / 4));
    }

    const verticalGrid = xTicks.map((tick) => `
      <line x1="${xFor(tick)}" y1="${margin.top}" x2="${xFor(tick)}" y2="${height - margin.bottom}" stroke="#e6ecf5" stroke-width="1" />
      <text x="${xFor(tick)}" y="${height - margin.bottom + 24}" text-anchor="middle" fill="#5c6b84" font-size="12">${tick}%</text>
    `).join("");

    const horizontalGrid = yTicks.map((tick) => `
      <line x1="${margin.left}" y1="${yFor(tick)}" x2="${width - margin.right}" y2="${yFor(tick)}" stroke="#e6ecf5" stroke-width="1" />
      <text x="${margin.left - 12}" y="${yFor(tick) + 4}" text-anchor="end" fill="#5c6b84" font-size="12">${formatInt(tick)}</text>
    `).join("");

    return `
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Speed investment versus max pnl curve" style="width: 100%; height: auto; display: block;">
        <rect x="0" y="0" width="${width}" height="${height}" rx="20" fill="#ffffff"></rect>
        ${verticalGrid}
        ${horizontalGrid}
        <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#24324a" stroke-width="1.25" />
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#24324a" stroke-width="1.25" />
        <polyline points="${points}" fill="none" stroke="#1b7f8a" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></polyline>
        <circle cx="${xFor(bestPoint.speedPct)}" cy="${yFor(bestPoint.pnl)}" r="6.5" fill="#ff7a59"></circle>
        <text x="${xFor(bestPoint.speedPct)}" y="${yFor(bestPoint.pnl) - 14}" text-anchor="middle" fill="#a43d26" font-size="12" font-weight="700">Best ${bestPoint.speedPct}%</text>
        <circle cx="${xFor(currentPoint.speedPct)}" cy="${yFor(currentPoint.pnl)}" r="6.5" fill="#2450d3"></circle>
        <text x="${xFor(currentPoint.speedPct)}" y="${yFor(currentPoint.pnl) + 22}" text-anchor="middle" fill="#1c3f9c" font-size="12" font-weight="700">You ${currentPoint.speedPct}%</text>
        <text x="${width / 2}" y="18" text-anchor="middle" fill="#152033" font-size="15" font-weight="700">Speed investment % against max PnL</text>
        <text x="${width / 2}" y="${height - 12}" text-anchor="middle" fill="#5c6b84" font-size="12">Speed investment (% of total budget)</text>
        <text x="18" y="${height / 2}" transform="rotate(-90 18 ${height / 2})" text-anchor="middle" fill="#5c6b84" font-size="12">Max PnL</text>
      </svg>
    `;
  }

  function buildDistributionChart(investments, userInvestment) {
    const counts = buildCounts([...investments, userInvestment]);
    const width = 980;
    const height = 200;
    const margin = { top: 18, right: 16, bottom: 34, left: 24 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const maxCount = Math.max(...counts, 1);

    function xFor(index) {
      return margin.left + innerWidth * index / counts.length;
    }

    function barWidth() {
      return innerWidth / counts.length - 1.5;
    }

    function yFor(value) {
      return margin.top + innerHeight * (1 - value / maxCount);
    }

    const bars = counts.map((count, index) => {
      const x = xFor(index);
      const y = yFor(count);
      const heightValue = height - margin.bottom - y;
      const isUserBucket = index === userInvestment;
      return `
        <rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${Math.max(1, barWidth()).toFixed(2)}" height="${Math.max(0, heightValue).toFixed(2)}" rx="2" fill="${isUserBucket ? "#2450d3" : "#9ed3d8"}"></rect>
      `;
    }).join("");

    return `
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Current simulated investment distribution" style="width: 100%; height: auto; display: block;">
        <rect x="0" y="0" width="${width}" height="${height}" rx="16" fill="#ffffff"></rect>
        <text x="${width / 2}" y="16" text-anchor="middle" fill="#152033" font-size="14" font-weight="700">Current field distribution</text>
        ${bars}
        <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#24324a" stroke-width="1" />
        <text x="${width / 2}" y="${height - 8}" text-anchor="middle" fill="#5c6b84" font-size="11">Speed investment (% of budget)</text>
      </svg>
    `;
  }

  function buildTopTable(points) {
    const rows = points.map((point, index) => `
      <tr>
        <td>${index + 1}</td>
        <td>${point.speedPct}%</td>
        <td>${formatInt(point.pnl)}</td>
        <td>${point.rank}</td>
        <td>${point.speedMultiplier.toFixed(3)}</td>
        <td>${point.researchPct}% / ${point.scalePct}%</td>
      </tr>
    `).join("");

    return `
      <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
        <thead>
          <tr>
            <th>#</th>
            <th>Speed</th>
            <th>Max PnL</th>
            <th>Rank</th>
            <th>Multiplier</th>
            <th>Best R / S</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  function buildMostCommonSpeedsTable(points, totalPlayers) {
    const rows = points.map((point, index) => `
      <tr>
        <td>${index + 1}</td>
        <td>${point.speedPct}%</td>
        <td>${point.count}</td>
        <td>${(100 * point.count / totalPlayers).toFixed(1)}%</td>
      </tr>
    `).join("");

    return `
      <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
        <thead>
          <tr>
            <th>#</th>
            <th>Speed</th>
            <th>Players</th>
            <th>Share</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  function buildHistoryMarkup(history) {
    const runKeys = Object.keys(history).sort((left, right) => {
      const leftValue = Number(left.split("_")[1]);
      const rightValue = Number(right.split("_")[1]);
      return leftValue - rightValue;
    });

    return runKeys.map((runKey) => {
      const iterationKeys = Object.keys(history[runKey]).sort((left, right) => {
        const leftValue = Number(left.split("_")[1]);
        const rightValue = Number(right.split("_")[1]);
        return leftValue - rightValue;
      });

      const iterationMarkup = iterationKeys.map((iterationKey) => {
        const record = history[runKey][iterationKey];
        return `
          <div style="padding: 6px 0; border-top: 1px solid #e6ecf5;">
            <strong>${runKey} ${iterationKey}</strong>
            <span style="color: #5c6b84;"> | user ${record.userInvestment}% | best ${record.bestSpeedPct}% | max pnl ${formatInt(record.bestPnl)}</span>
          </div>
        `;
      }).join("");

      return `
        <div style="padding: 10px 12px; background: #ffffff; border: 1px solid #d9e4ef; border-radius: 14px;">
          <div style="font-weight: 700; margin-bottom: 6px;">${runKey}</div>
          ${iterationMarkup}
        </div>
      `;
    }).join("");
  }

  function storeCurrentIteration() {
    const runKey = `run_${state.runNumber}`;
    const iterationKey = `iteration_${state.iterationNumber}`;

    if (!state.history[runKey]) {
      state.history[runKey] = {};
    }

    const curve = computeCurve(state.simulatedInvestments);
    const currentOutcome = curve.find((point) => point.speedPct === state.userInvestment) || curve[0];
    const bestOutcome = computeTopCurvePoints(curve, 1)[0];

    state.curve = curve;
    state.currentOutcome = currentOutcome;
    state.bestOutcome = bestOutcome;
    state.history[runKey][iterationKey] = {
      userInvestment: state.userInvestment,
      bestSpeedPct: bestOutcome.speedPct,
      bestPnl: bestOutcome.pnl,
      curve: curve.map((point) => ({
        speedPct: point.speedPct,
        pnl: point.pnl,
        rank: point.rank,
        speedMultiplier: Number(point.speedMultiplier.toFixed(12)),
        researchPct: point.researchPct,
        scalePct: point.scalePct,
      })),
    };
  }

  function render() {
    storeCurrentIteration();
    const topPoints = computeTopCurvePoints(state.curve, 8);
    const mostCommonSpeeds = computeMostCommonSpeeds(state.simulatedInvestments, state.userInvestment, 8);

    root.innerHTML = `
      <style>
        #${containerId} * { box-sizing: border-box; }
        #${containerId} {
          font-family: Arial, sans-serif;
          color: #152033;
        }
        #${containerId} .sim-shell {
          background: linear-gradient(180deg, #f7fbff 0%, #eef4fa 100%);
          border: 1px solid #d9e4ef;
          border-radius: 22px;
          padding: 22px;
        }
        #${containerId} .sim-header {
          display: flex;
          gap: 16px;
          justify-content: space-between;
          align-items: flex-start;
          flex-wrap: wrap;
          margin-bottom: 20px;
        }
        #${containerId} .sim-title {
          font-size: 1.35rem;
          font-weight: 800;
          margin: 0 0 6px 0;
        }
        #${containerId} .sim-subtitle {
          color: #42536f;
          line-height: 1.45;
          max-width: 820px;
        }
        #${containerId} .sim-actions {
          display: flex;
          gap: 10px;
          align-items: center;
          flex-wrap: wrap;
        }
        #${containerId} .sim-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 12px;
          margin-bottom: 18px;
        }
        #${containerId} .sim-card {
          background: #ffffff;
          border: 1px solid #d9e4ef;
          border-radius: 16px;
          padding: 14px 16px;
        }
        #${containerId} .sim-card-label {
          color: #5c6b84;
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          margin-bottom: 6px;
        }
        #${containerId} .sim-card-value {
          font-size: 1.18rem;
          font-weight: 800;
        }
        #${containerId} .sim-row {
          display: grid;
          grid-template-columns: minmax(0, 2.1fr) minmax(300px, 1fr);
          gap: 18px;
          margin-bottom: 18px;
        }
        #${containerId} .sim-panel {
          background: #ffffff;
          border: 1px solid #d9e4ef;
          border-radius: 18px;
          padding: 16px;
        }
        #${containerId} .sim-panel-title {
          font-size: 1rem;
          font-weight: 800;
          margin-bottom: 10px;
        }
        #${containerId} table th,
        #${containerId} table td {
          text-align: left;
          padding: 8px 6px;
          border-bottom: 1px solid #eef3f8;
        }
        #${containerId} table th {
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: #5c6b84;
        }
        #${containerId} .sim-input-wrap {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
          align-items: center;
          margin-top: 6px;
        }
        #${containerId} input[type="number"] {
          width: 140px;
          border: 1px solid #b8c7d9;
          border-radius: 12px;
          padding: 10px 12px;
          font-size: 14px;
        }
        #${containerId} button {
          border: none;
          border-radius: 12px;
          padding: 10px 14px;
          font-size: 14px;
          font-weight: 700;
          cursor: pointer;
        }
        #${containerId} .primary-btn {
          background: #2450d3;
          color: #ffffff;
        }
        #${containerId} .secondary-btn {
          background: #dde7f8;
          color: #16327d;
        }
        #${containerId} .sim-note {
          margin-top: 10px;
          font-size: 13px;
          color: #42536f;
          line-height: 1.45;
        }
        #${containerId} .sim-meta {
          display: flex;
          gap: 12px;
          flex-wrap: wrap;
          color: #42536f;
          font-size: 13px;
          margin-top: 6px;
        }
        #${containerId} .sim-mini-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
          gap: 14px;
          margin-top: 16px;
        }
        @media (max-width: 950px) {
          #${containerId} .sim-row {
            grid-template-columns: 1fr;
          }
        }
      </style>
      <div class="sim-shell">
        <div class="sim-header">
          <div>
            <div class="sim-title">Round 2 manual-trading speed simulator</div>
            <div class="sim-subtitle">
              The field contains 999 simulated players plus you. Each curve point inserts a hypothetical you at that speed and evaluates max PnL with the same research and scale logic as the round-2 notebook.
            </div>
            <div class="sim-meta">
              <span><strong>Stored:</strong> run_${state.runNumber} iteration_${state.iterationNumber}</span>
              <span><strong>History access:</strong> window.manualTradingRound2SimulatorRegistry["${containerId}"].getHistory()</span>
            </div>
          </div>
          <div class="sim-actions">
            <button type="button" class="secondary-btn" data-action="new-run">Start new run</button>
          </div>
        </div>

        <div class="sim-grid">
          <div class="sim-card">
            <div class="sim-card-label">Your speed</div>
            <div class="sim-card-value">${formatPct(state.userInvestment)}</div>
          </div>
          <div class="sim-card">
            <div class="sim-card-label">Your rank</div>
            <div class="sim-card-value">${state.currentOutcome.rank}</div>
          </div>
          <div class="sim-card">
            <div class="sim-card-label">Your multiplier</div>
            <div class="sim-card-value">${state.currentOutcome.speedMultiplier.toFixed(3)}</div>
          </div>
          <div class="sim-card">
            <div class="sim-card-label">Your max PnL</div>
            <div class="sim-card-value">${formatInt(state.currentOutcome.pnl)}</div>
          </div>
          <div class="sim-card">
            <div class="sim-card-label">Best speed on curve</div>
            <div class="sim-card-value">${formatPct(state.bestOutcome.speedPct)}</div>
          </div>
          <div class="sim-card">
            <div class="sim-card-label">Next noise scale</div>
            <div class="sim-card-value">${Math.max(0, state.currentErrorStd).toFixed(1)}</div>
          </div>
        </div>

        <div class="sim-row">
          <div class="sim-panel">
            <div class="sim-panel-title">Curve for the current field</div>
            ${buildLineChart(state.curve, state.userInvestment)}
            <div class="sim-note">
              The first curve is stored as <strong>run_1 iteration_1</strong>. On each submit, other players best-respond against your latest speed, then move to one random optimum plus Laplace noise.
            </div>
          </div>
          <div class="sim-panel">
            <div class="sim-panel-title">Advance the simulation</div>
            <div style="color: #42536f; line-height: 1.5;">
              Enter your next speed investment. The 999 simulated players will consult the current best-response lookup, with ties broken randomly. Starting from the second update, a random number of players are frozen each round with Gaussian mean 10 and variance 3.
            </div>
            <div class="sim-input-wrap">
              <input type="number" min="${config.minInvestment}" max="${config.maxInvestment}" step="1" value="${state.userInvestment}" data-role="user-input" />
              <button type="button" class="primary-btn" data-action="submit-speed">Submit next investment</button>
            </div>
            <div class="sim-note" data-role="status">
              ${state.lastStatus}
            </div>
            <div class="sim-note">
              Your current best allocation at ${formatPct(state.userInvestment)} is research <strong>${formatPct(state.currentOutcome.researchPct)}</strong>, scale <strong>${formatPct(state.currentOutcome.scalePct)}</strong>, speed <strong>${formatPct(state.userInvestment)}</strong>.
            </div>
            <div class="sim-note">
              Last moving round: frozen players = <strong>${state.lastFrozenCount}</strong>, noise scale used = <strong>${state.lastUsedErrorStd.toFixed(1)}</strong>.
            </div>
            <div class="sim-mini-grid">
              <div>
                <div class="sim-panel-title">Top curve points</div>
                ${buildTopTable(topPoints)}
              </div>
              <div>
                <div class="sim-panel-title">Most common speeds</div>
                ${buildMostCommonSpeedsTable(mostCommonSpeeds, config.totalPlayers)}
              </div>
            </div>
          </div>
        </div>

        <div class="sim-row" style="align-items: start;">
          <div class="sim-panel">
            <div class="sim-panel-title">Field distribution after this iteration</div>
            ${buildDistributionChart(state.simulatedInvestments, state.userInvestment)}
          </div>
          <div class="sim-panel">
            <div class="sim-panel-title">Stored iteration history</div>
            ${buildHistoryMarkup(state.history)}
          </div>
        </div>
      </div>
    `;

    root.querySelector('[data-action="submit-speed"]').addEventListener("click", handleSubmit);
    root.querySelector('[data-action="new-run"]').addEventListener("click", startNewRun);
  }

  function startNewRun() {
    state.runNumber += 1;
    state.iterationNumber = 1;
    state.currentErrorStd = config.initialErrorStd;
    state.lastUsedErrorStd = 0;
    state.lastFrozenCount = 0;
    state.simulatedInvestments = Array.from(
      { length: config.simulatedPlayers },
      () => randomInteger(config.minInvestment, config.maxInvestment),
    );
    state.userInvestment = randomInteger(config.minInvestment, config.maxInvestment);
    state.lastStatus = `New run started. The initial curve for run_${state.runNumber} iteration_1 is ready, and your random starting speed is ${state.userInvestment}%.`;
    render();
  }

  function handleSubmit() {
    const input = root.querySelector('[data-role="user-input"]');
    const numericValue = Number(input.value);

    if (!Number.isFinite(numericValue)) {
      state.lastStatus = `Enter an integer speed between ${config.minInvestment}% and ${config.maxInvestment}%.`;
      render();
      return;
    }

    const submittedInvestment = clamp(Math.round(numericValue), config.minInvestment, config.maxInvestment);
    const lookup = buildLookup(state.simulatedInvestments, submittedInvestment);
    const frozenCount = state.iterationNumber >= 2 ? sampleFrozenCount() : 0;
    const frozenIndices = randomIndexSet(config.simulatedPlayers, frozenCount);
    const nextInvestments = state.simulatedInvestments.slice();
    const usedStd = state.currentErrorStd;

    state.userInvestment = submittedInvestment;

    for (let index = 0; index < nextInvestments.length; index += 1) {
      if (frozenIndices.has(index)) {
        continue;
      }

      const currentInvestment = nextInvestments[index];
      const bestTargets = lookup[currentInvestment].bestTargets;
      const chosenTarget = randomChoice(bestTargets);
      const noisyTarget = Math.round(laplace(chosenTarget, usedStd));
      nextInvestments[index] = clamp(noisyTarget, config.minInvestment, config.maxInvestment);
    }

    state.simulatedInvestments = nextInvestments;
    state.iterationNumber += 1;
    state.lastFrozenCount = frozenCount;
    state.lastUsedErrorStd = usedStd;
    state.currentErrorStd = Math.max(0, state.currentErrorStd - config.errorStdDecay);
    state.lastStatus = `Stored run_${state.runNumber} iteration_${state.iterationNumber}. You moved to ${submittedInvestment}%, ${config.simulatedPlayers - frozenCount} simulated players updated, and ${frozenCount} stayed fixed.`;
    render();
  }

  const state = {
    runNumber: 1,
    iterationNumber: 1,
    currentErrorStd: config.initialErrorStd,
    lastUsedErrorStd: 0,
    lastFrozenCount: 0,
    simulatedInvestments: Array.from(
      { length: config.simulatedPlayers },
      () => randomInteger(config.minInvestment, config.maxInvestment),
    ),
    userInvestment: randomInteger(config.minInvestment, config.maxInvestment),
    history: {},
    curve: [],
    currentOutcome: null,
    bestOutcome: null,
    lastStatus: "",
  };

  registry[containerId] = {
    getHistory: () => JSON.parse(JSON.stringify(state.history)),
    getSnapshot: () => JSON.parse(JSON.stringify({
      runNumber: state.runNumber,
      iterationNumber: state.iterationNumber,
      userInvestment: state.userInvestment,
      currentErrorStd: state.currentErrorStd,
      lastFrozenCount: state.lastFrozenCount,
      lastUsedErrorStd: state.lastUsedErrorStd,
      simulatedInvestments: state.simulatedInvestments,
    })),
  };

  state.lastStatus = `Initialised run_1 iteration_1. Your random starting speed is ${state.userInvestment}%, and the curve below is ready for your next input.`;
  render();
})();
</script>
"""
    return (
        template.replace("__CONTAINER_ID__", json.dumps(container_id))
        .replace("__CONFIG_JSON__", json.dumps(config))
    )


def display_round2_manual_trading_simulator(container_id: str | None = None) -> None:
    """Display the simulator inside a Jupyter notebook."""
    display(HTML(build_round2_manual_trading_simulator_html(container_id=container_id)))
