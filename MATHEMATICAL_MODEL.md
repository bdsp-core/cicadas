# Mathematical Model for ICU EEG PKPD Emulated Trial

This document describes the complete mathematical and statistical model that generates the synthetic clinical trial data. This model enables causal inference studies comparing randomized controlled trials (RCT = 1) versus observational studies (RCT = 0).

## Overview

The model simulates patient-level trajectories over 168 hours (7 days) with 0.5-hour time steps, incorporating:
- Patient-specific disease and treatment harm parameters
- Disease natural history with log-normal trajectory
- PI-controller based treatment administration  
- Time-dependent mortality hazard
- Treatment assignment mechanisms (randomized vs observational)

## 1. Patient-Level Parameters

Each patient $i$ is characterized by:

### 1.1 Harm Parameters
- **Disease harm**: $\text{harmE}_i \sim \max(0, \mathcal{N}(30, 10^2))$ (truncated normal)
- **Treatment harm**: $\text{harmA}_i \sim \max(0, \mathcal{N}(10, 5^2))$ (truncated normal)

### 1.2 Disease Natural History Parameters
Each patient has individualized pulse parameters with ±10-20% variation:
- **Pulse amplitude**: $\text{pulseAmp}_i \sim \max(0, \mathcal{N}(1, (0.1)^2))$
- **Pulse timing**: $\text{pulseMu}_i \sim \max(0, \mathcal{N}(20, (4)^2))$  
- **Pulse width**: $\text{pulseWidth}_i \sim \max(0, \mathcal{N}(1.8, (0.36)^2))$
- **Pulse center**: $\text{pulseC}_i \sim \max(0, \mathcal{N}(15, (3)^2))$

## 2. Treatment Assignment

### 2.1 Randomized Controlled Trial (RCT = 1)
$$P(\text{Rx}_i = 1) = 0.5$$

Treatment assignment is independent of patient characteristics.

### 2.2 Observational Study (RCT = 0)  
$$P(\text{Rx}_i = 1) = 0.5 + 0.2 \cdot \frac{\text{harmE}_i - \text{harmA}_i}{\text{harmE}_i + \text{harmA}_i}$$

Treatment probability depends on the relative harm parameters, creating selection bias where patients with higher disease-to-treatment harm ratios are more likely to receive treatment.

## 3. Disease Progression Model

### 3.1 Natural History Function
The disease natural history follows a log-normal shaped trajectory:

$$f_i(t) = \frac{\text{pulseAmp}_i}{\max(g_i(t))} \cdot g_i(t)$$

where:
$$g_i(t) = t^{\text{pulseC}_i} \cdot \exp\left(-\frac{\text{pulseC}_i \cdot t}{\text{pulseMu}_i \cdot \text{pulseWidth}_i}\right)$$

### 3.2 Treatment Effect Function
The treatment suppression factor is:
$$s_A(A_{i,j}) = 1 - \frac{1}{(C/A_{i,j})^g + 1}$$

where:
- $C = 3$ (drug potency)
- $g = 4$ (dose-response steepness)
- $A_{i,j}$ is the treatment amount for patient $i$ at time $j$

### 3.3 Observed Disease Burden
$$L_{i,j} = (b_0 + f_i(t_j)) \cdot s_A(A_{i,j-1})$$

where $b_0 = 0.1$ is the baseline disease level.

## 4. Treatment Control System

### 4.1 PI Controller
The treatment is determined by a proportional-integral controller:

$$e_{i,j} = L_{i,j} - \theta$$

$$\text{eInt}_{i,j} = \text{eInt}_{i,j-1} + e_{i,j} \cdot \Delta t$$

$$A_{\text{unsat},i,j} = k_p \cdot e_{i,j} + k_i \cdot \text{eInt}_{i,j}$$

where:
- $\theta = 0.05$ (target threshold)
- $k_p = 0$ (proportional gain - pure integral control)  
- $k_i = 20$ (integral gain)
- $\Delta t = 0.5$ hours

### 4.2 Treatment Saturation and Anti-Windup
$$A_{i,j} = \text{Rx}_{i} \cdot \min(\max(A_{\text{unsat},i,j}, 0), A_{\max})$$

where $A_{\max} = 50$.

Anti-windup: If $A_{i,j} \neq A_{\text{unsat},i,j}$:
$$\text{eInt}_{i,j} = \text{eInt}_{i,j} - \frac{A_{\text{unsat},i,j} - A_{i,j}}{k_i}$$

## 5. Mortality Hazard Model

### 5.1 Hazard Components
The instantaneous mortality hazard has two components:

**Disease/Treatment Component:**
$$h_{1,i}(t_j) = \text{scale}_1 \cdot \left[b_0 + \text{harmE}_i \cdot \frac{\sum_{k=1}^j L_{i,k}}{18.4} + \text{harmA}_i \cdot \frac{\sum_{k=1}^j A_{i,k}}{360}\right]$$

**Time Component:**
$$h_{2,i}(t_j) = \text{scale}_2 \cdot \left(\frac{j}{N_t-1}\right)^2$$

where:
- $\text{scale}_1 = 0.0005$ 
- $\text{scale}_2 = 0.0025$
- $N_t = 337$ (total time points)

### 5.2 Total Hazard
$$h_{Y,i}(t_j) = h_{1,i}(t_j) + h_{2,i}(t_j)$$

### 5.3 Survival Probability
$$S_{Y,i}(t_j) = S_{Y,i}(t_{j-1}) \cdot \exp(-h_{Y,i}(t_j) \cdot \Delta t)$$

### 5.4 Death Event
$$Y_{i,j} \sim \text{Bernoulli}(h_{Y,i}(t_j) \cdot \Delta t)$$

Once $Y_{i,j} = 1$, the patient exits the study: $Y_{i,k} = 1$ for all $k \geq j$.

## 6. Observational Study Complications (RCT = 0 only)

### 6.1 Treatment Switching
Exponentially decreasing switch hazard:
$$h_{\text{switch}}(t) = 0.015 \cdot \exp(-0.03 \cdot t)$$

Treatment switches occur with probability $h_{\text{switch}}(t_j) \cdot \Delta t$.

### 6.2 Treatment Stopping  
Treatment stopping hazard (for treated patients):
$$h_{\text{stop}}(L_{i,j}) = 0.02 \cdot \exp(-5 \cdot L_{i,j})$$

Higher when disease burden is low (patients stop when doing well).

### 6.3 Dropout Hazard
$$h_{\text{dropout},i}(t_j) = 0.001 \cdot \left[1 + 2L_{i,j}^2 + 0.5(\text{harmE}_i + \text{harmA}_i) + 10\left(\frac{j}{N_t}\right)^2\right]$$

Depends on current disease burden, patient harm parameters, and time.

## 7. Key Model Features for Causal Inference

### 7.1 Confounding Structure
In observational studies (RCT = 0):
- **Treatment selection bias**: $P(\text{Rx} = 1 | \text{harmE}, \text{harmA})$ depends on harm parameters
- **Time-varying confounding**: Treatment switching depends on evolving disease burden
- **Informative censoring**: Dropout depends on disease trajectory and patient characteristics

### 7.2 Identifiability Conditions
The model satisfies key assumptions for causal identification:
- **No unmeasured confounders**: All confounders ($\text{harmE}_i$, $\text{harmA}_i$, $L_{i,j}$) are observed
- **Positivity**: $0 < P(\text{Rx} = 1 | \text{harmE}, \text{harmA}) < 1$ for all patients
- **Consistency**: Treatment effects are deterministic given patient state
- **Sequential ignorability**: Treatment at time $t$ is independent of future potential outcomes given observed history

### 7.3 True Causal Estimand
The target estimand is the **Average Treatment Effect on Survival** from the RCT setting:
$$\text{ATE} = E[Y^1] - E[Y^0]$$

where $Y^a$ is the potential survival outcome under treatment regime $a$.

## 8. Implementation Notes

### 8.1 Time Discretization
- Time steps: $t_j = j \cdot \Delta t$ for $j = 0, 1, \ldots, 336$ 
- $\Delta t = 0.5$ hours
- Total follow-up: 168 hours (7 days)

### 8.2 Early Termination
Simulation terminates early when:
- Death occurs: $Y_{i,j} = 1$
- Dropout occurs: $V_{i,j} = 1$ (RCT = 0 only)

### 8.3 Numerical Stability
- All random parameters are truncated at 0 to ensure positivity
- PI controller includes anti-windup mechanism
- Hazard calculations use small time steps to maintain accuracy

## 9. Data Output Structure

Each patient contributes a longitudinal dataset with variables:
- **Patient ID**: $\text{sid}_i$
- **Time points**: $t_j$  
- **Treatment assignment**: $\text{Rx}_i$
- **Harm parameters**: $\text{harmE}_i$, $\text{harmA}_i$
- **Disease burden**: $L_{i,j}$
- **Treatment amount**: $A_{i,j}$
- **Death indicator**: $Y_{i,j}$
- **Pulse parameters**: $\text{pulseAmp}_i$, $\text{pulseMu}_i$, $\text{pulseWidth}_i$, $\text{pulseC}_i$

This longitudinal structure enables sophisticated causal inference methods including g-methods, targeted maximum likelihood estimation (TMLE), and double-robust estimation.