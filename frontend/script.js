// REAL ESTATE AI — Enterprise Valuation & Market Intelligence Platform Engine
// Client-Side Application State
const appState = {
    propertyInput: null,
    prediction: null,
    explainability: null,
    marketComparison: null,
    overviewMetrics: null,
    featuresMeta: null,
    globalExplainability: null
};

const STORAGE_KEY = 'real_estate_ai_platform_state';

// Lifecycle Bootstrap
document.addEventListener('DOMContentLoaded', async () => {
    // 1. Navigation Routing Setup
    initNavigation();

    // 2. Range Sliders Display Listeners
    initSliders();

    // 3. Load Overview Metrics from GET /overview-metrics
    await loadOverviewMetrics();

    // 4. Load Dynamic Form Choices from GET /features
    await loadFeatureOptions();

    // 5. Load Global Explainability Metadata
    await loadGlobalExplainability();

    // 6. Form Submission Listener
    initValuationForm();

    // 7. Setup What-If Controls
    initWhatIfControls();

    // 8. Restore Session State if available
    restoreSessionState();
});

// ----------------------------------------------------
// ROUTING & NAVIGATION
// ----------------------------------------------------
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            const pageId = item.getAttribute('data-page');
            if (pageId) {
                e.preventDefault();
                navigateTo(pageId);
            }
        });
    });

    // Check URL Hash on Load
    const initialHash = window.location.hash.replace('#', '').toLowerCase();
    if (initialHash === 'analytics' || initialHash === 'model-analytics' || initialHash === 'system' || initialHash === 'about') {
        navigateTo('page-overview');
    } else if (initialHash) {
        const matchingNav = document.querySelector(`.nav-item[href="#${initialHash}"]`);
        if (matchingNav) {
            const pageId = matchingNav.getAttribute('data-page');
            navigateTo(pageId);
        } else {
            navigateTo('page-overview');
        }
    }
}

function navigateTo(pageId) {
    if (pageId === 'page-analytics' || pageId === 'page-system' || pageId === 'page-about') {
        pageId = 'page-overview';
    }
    const pages = document.querySelectorAll('.page-view');
    const navItems = document.querySelectorAll('.nav-item');

    pages.forEach(page => page.classList.remove('active'));
    navItems.forEach(item => item.classList.remove('active'));

    const targetPage = document.getElementById(pageId) || document.getElementById('page-overview');
    if (targetPage) {
        targetPage.classList.add('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });

        const activeNav = document.querySelector(`.nav-item[data-page="${targetPage.id}"]`);
        if (activeNav) {
            activeNav.classList.add('active');
            const href = activeNav.getAttribute('href');
            if (href && history.pushState) {
                history.pushState(null, null, href);
            }
        }
    }
}

// ----------------------------------------------------
// SLIDERS REAL-TIME LABELS
// ----------------------------------------------------
function initSliders() {
    const distMetro = document.getElementById('dist_metro');
    const distMetroVal = document.getElementById('dist_metro_val');
    const distCity = document.getElementById('dist_city');
    const distCityVal = document.getElementById('dist_city_val');

    if (distMetro && distMetroVal) {
        distMetro.addEventListener('input', (e) => {
            distMetroVal.textContent = parseFloat(e.target.value).toFixed(1);
        });
    }

    if (distCity && distCityVal) {
        distCity.addEventListener('input', (e) => {
            distCityVal.textContent = parseFloat(e.target.value).toFixed(1);
        });
    }
}

// ----------------------------------------------------
// OVERVIEW METRICS LOADER (GET /overview-metrics)
// ----------------------------------------------------
async function loadOverviewMetrics() {
    try {
        const response = await fetch('/overview-metrics');
        if (!response.ok) throw new Error('Failed to load overview metrics');

        const data = await response.json();
        appState.overviewMetrics = data;

        const kpiProps = document.getElementById('kpi-props');
        if (kpiProps) kpiProps.textContent = `${data.properties_analyzed.toLocaleString()}`;

        const kpiCities = document.getElementById('kpi-cities');
        if (kpiCities) kpiCities.textContent = `${data.cities_covered} Cities`;

        const kpiAvg = document.getElementById('kpi-avg-price');
        if (kpiAvg) kpiAvg.textContent = `₹ ${data.average_price_lakhs.toFixed(2)} L`;

        const kpiMed = document.getElementById('kpi-med-price');
        if (kpiMed) kpiMed.textContent = `₹ ${data.median_price_lakhs.toFixed(2)} L`;

        const kpiR2 = document.getElementById('kpi-r2');
        if (kpiR2) kpiR2.textContent = data.model_r2.toFixed(4);

        const kpiMae = document.getElementById('kpi-mae');
        if (kpiMae) kpiMae.textContent = `₹ ${data.model_mae_lakhs.toFixed(2)} L`;

        const snapCity = document.getElementById('snap-top-city');
        if (snapCity && data.top_city_by_median) {
            snapCity.textContent = `${data.top_city_by_median.city} (₹ ${data.top_city_by_median.median_price_lakhs.toFixed(2)} L)`;
        }

        const snapFeat = document.getElementById('snap-top-feat');
        if (snapFeat && data.most_important_feature) {
            snapFeat.textContent = `${data.most_important_feature.feature} (${data.most_important_feature.importance_percentage}%)`;
        }

        const sidebarStatus = document.getElementById('sidebar-status-text');
        if (sidebarStatus) sidebarStatus.textContent = 'System Online';

    } catch (err) {
        console.warn('Overview metrics warning:', err);
        const sidebarStatus = document.getElementById('sidebar-status-text');
        if (sidebarStatus) sidebarStatus.textContent = 'Offline';
    }
}

// ----------------------------------------------------
// FEATURE CHOICES LOADER (GET /features)
// ----------------------------------------------------
async function loadFeatureOptions() {
    try {
        const response = await fetch('/features');
        if (!response.ok) throw new Error('Failed to load feature choices');

        const data = await response.json();
        appState.featuresMeta = data;

        populateSelect('city', data.cities, data.defaults.City);
        populateSelect('locality_type', data.locality_types, data.defaults.Locality_Type);
        populateSelect('property_type', data.property_types, data.defaults.Property_Type);
        populateSelect('furnishing_status', data.furnishing_statuses, data.defaults.Furnishing_Status);

    } catch (err) {
        showToast('⚠️ Unable to connect to feature options service.');
        console.error('Feature metadata fetch error:', err);
    }
}

function populateSelect(elementId, options, defaultValue) {
    const select = document.getElementById(elementId);
    if (!select || !options) return;

    select.innerHTML = '';
    options.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt;
        option.textContent = opt;
        if (opt === defaultValue) option.selected = true;
        select.appendChild(option);
    });
}

function resetFormToDefaults() {
    if (!appState.featuresMeta || !appState.featuresMeta.defaults) return;
    const def = appState.featuresMeta.defaults;

    document.getElementById('city').value = def.City;
    document.getElementById('locality_type').value = def.Locality_Type;
    document.getElementById('property_type').value = def.Property_Type;
    document.getElementById('furnishing_status').value = def.Furnishing_Status;

    document.getElementById('bhk').value = def.BHK;
    document.getElementById('bathrooms').value = def.Bathrooms;
    document.getElementById('super_area').value = def.Super_Area_SqFt;
    document.getElementById('carpet_area').value = def.Carpet_Area_SqFt;
    document.getElementById('floor_number').value = def.Floor_Number;
    document.getElementById('total_floors').value = def.Total_Floors;
    document.getElementById('age').value = def.Age_of_Property;

    document.getElementById('parking').checked = def.Parking === 1;
    document.getElementById('lift').checked = def.Lift_Available === 1;
    document.getElementById('gated').checked = def.Gated_Community === 1;

    document.getElementById('dist_metro').value = def.Distance_to_Metro_km;
    document.getElementById('dist_metro_val').textContent = def.Distance_to_Metro_km.toFixed(1);
    document.getElementById('dist_city').value = def.Distance_to_City_Center_km;
    document.getElementById('dist_city_val').textContent = def.Distance_to_City_Center_km.toFixed(1);

    showToast('Form reset to standard property defaults.');
}

// ----------------------------------------------------
// FORM SUBMISSION & MODEL EXECUTION
// ----------------------------------------------------
function initValuationForm() {
    const form = document.getElementById('valuation-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const btn = document.getElementById('analyze-btn');
        btn.textContent = '⏳ Predicting Property Value...';
        btn.disabled = true;

        try {
            const superArea = parseFloat(document.getElementById('super_area').value);
            const carpetArea = parseFloat(document.getElementById('carpet_area').value);
            const floorNumber = parseInt(document.getElementById('floor_number').value, 10);
            const totalFloors = parseInt(document.getElementById('total_floors').value, 10);
            const bhk = parseInt(document.getElementById('bhk').value, 10);
            const bathrooms = parseInt(document.getElementById('bathrooms').value, 10);
            const age = parseInt(document.getElementById('age').value, 10);
            const city = document.getElementById('city').value;
            const localityType = document.getElementById('locality_type').value;
            const propertyType = document.getElementById('property_type').value;
            const furnishingStatus = document.getElementById('furnishing_status').value;

            // Strict Client-Side Input Validation
            if (!city || !localityType || !propertyType || !furnishingStatus) {
                showToast('⚠️ Please select all required categorical property options.');
                btn.textContent = '⚡ PREDICT PROPERTY VALUE';
                btn.disabled = false;
                return;
            }

            if (isNaN(superArea) || superArea <= 0) {
                showToast('⚠️ Please enter a valid Super Built-up Area.');
                btn.textContent = '⚡ PREDICT PROPERTY VALUE';
                btn.disabled = false;
                return;
            }

            if (isNaN(carpetArea) || carpetArea <= 0) {
                showToast('⚠️ Please enter a valid Carpet Area.');
                btn.textContent = '⚡ PREDICT PROPERTY VALUE';
                btn.disabled = false;
                return;
            }

            if (carpetArea > superArea) {
                showToast('⚠️ Carpet Area cannot exceed Super Built-up Area.');
                btn.textContent = '⚡ PREDICT PROPERTY VALUE';
                btn.disabled = false;
                return;
            }

            if (floorNumber > totalFloors) {
                showToast('⚠️ Floor Number cannot exceed Total Floors in building.');
                btn.textContent = '⚡ PREDICT PROPERTY VALUE';
                btn.disabled = false;
                return;
            }

            if (floorNumber < 0 || totalFloors < 1 || age < 0 || bhk < 1 || bathrooms < 1) {
                showToast('⚠️ Please ensure floor, age, and room counts are positive valid numbers.');
                btn.textContent = '⚡ PREDICT PROPERTY VALUE';
                btn.disabled = false;
                return;
            }

            const propertyData = {
                City: city,
                Locality_Type: localityType,
                Property_Type: propertyType,
                Furnishing_Status: furnishingStatus,
                Super_Area_SqFt: superArea,
                Carpet_Area_SqFt: carpetArea,
                BHK: bhk,
                Bathrooms: bathrooms,
                Floor_Number: floorNumber,
                Total_Floors: totalFloors,
                Age_of_Property: age,
                Parking: document.getElementById('parking').checked ? 1 : 0,
                Lift_Available: document.getElementById('lift').checked ? 1 : 0,
                Gated_Community: document.getElementById('gated').checked ? 1 : 0,
                Distance_to_Metro_km: parseFloat(document.getElementById('dist_metro').value),
                Distance_to_City_Center_km: parseFloat(document.getElementById('dist_city').value)
            };

            // 1. Run Main Prediction (POST /predict)
            const predResponse = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(propertyData)
            });

            if (!predResponse.ok) {
                const errJson = await predResponse.json();
                throw new Error(errJson.detail || 'Prediction request failed on backend.');
            }

            const predResult = await predResponse.json();
            appState.propertyInput = propertyData;
            appState.prediction = predResult;

            // 2. Fetch Contextual Explainability (POST /explain)
            const expResponse = await fetch('/explain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(propertyData)
            });
            if (expResponse.ok) {
                appState.explainability = await expResponse.json();
            }

            // 3. Fetch Real Market Comparison (POST /market-comparison)
            const mktResponse = await fetch('/market-comparison', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(propertyData)
            });
            if (mktResponse.ok) {
                appState.marketComparison = await mktResponse.json();
            }

            // Save in localStorage
            saveSessionState();

            // Render All Views
            renderAllViews();

            showToast('✅ Property valuation generated successfully!');

            // Automatically Navigate to Prediction Result
            navigateTo('page-result');

        } catch (err) {
            showToast(`⚠️ API Error: ${err.message}`);
            console.error('Valuation error:', err);
        } finally {
            btn.textContent = '⚡ PREDICT PROPERTY VALUE';
            btn.disabled = false;
        }
    });
}

// ----------------------------------------------------
// STATE PERSISTENCE
// ----------------------------------------------------
function saveSessionState() {
    try {
        const payload = {
            propertyInput: appState.propertyInput,
            prediction: appState.prediction,
            explainability: appState.explainability,
            marketComparison: appState.marketComparison
        };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch (e) {
        console.warn('Unable to persist session state:', e);
    }
}

function restoreSessionState() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) {
            renderEmptyStates();
            return;
        }

        const parsed = JSON.parse(raw);
        if (parsed.propertyInput && parsed.prediction) {
            appState.propertyInput = parsed.propertyInput;
            appState.prediction = parsed.prediction;
            appState.explainability = parsed.explainability;
            appState.marketComparison = parsed.marketComparison;

            // Repopulate form inputs
            populateFormFromInput(parsed.propertyInput);

            // Render Views
            renderAllViews();
        } else {
            renderEmptyStates();
        }
    } catch (e) {
        console.warn('Session state restore error:', e);
        renderEmptyStates();
    }
}

function populateFormFromInput(inp) {
    if (!inp) return;
    const setVal = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.value = val;
    };

    setVal('city', inp.City);
    setVal('locality_type', inp.Locality_Type);
    setVal('property_type', inp.Property_Type);
    setVal('furnishing_status', inp.Furnishing_Status);
    setVal('super_area', inp.Super_Area_SqFt);
    setVal('carpet_area', inp.Carpet_Area_SqFt);
    setVal('bhk', inp.BHK);
    setVal('bathrooms', inp.Bathrooms);
    setVal('floor_number', inp.Floor_Number);
    setVal('total_floors', inp.Total_Floors);
    setVal('age', inp.Age_of_Property);

    const parking = document.getElementById('parking');
    if (parking) parking.checked = inp.Parking === 1;
    const lift = document.getElementById('lift');
    if (lift) lift.checked = inp.Lift_Available === 1;
    const gated = document.getElementById('gated');
    if (gated) gated.checked = inp.Gated_Community === 1;

    setVal('dist_metro', inp.Distance_to_Metro_km);
    const metroVal = document.getElementById('dist_metro_val');
    if (metroVal) metroVal.textContent = parseFloat(inp.Distance_to_Metro_km).toFixed(1);

    setVal('dist_city', inp.Distance_to_City_Center_km);
    const cityVal = document.getElementById('dist_city_val');
    if (cityVal) cityVal.textContent = parseFloat(inp.Distance_to_City_Center_km).toFixed(1);
}

// ----------------------------------------------------
// EMPTY STATES & VIEW DISPATCH
// ----------------------------------------------------
function renderEmptyStates() {
    // Prediction Result
    toggleVisibility('result-empty-state', true);
    toggleVisibility('result-content-view', false);

    // Explainable AI
    toggleVisibility('explain-empty-state', true);
    toggleVisibility('explain-content-view', false);

    // Market Intelligence
    toggleVisibility('market-empty-state', true);
    toggleVisibility('market-content-view', false);

    // AI Insights
    toggleVisibility('insights-empty-state', true);
    toggleVisibility('insights-content-view', false);

    // Overview Recent Valuation
    renderOverviewRecentValuation(null);
}

function toggleVisibility(elementId, isVisible) {
    const el = document.getElementById(elementId);
    if (!el) return;
    if (isVisible) {
        el.classList.remove('hidden');
    } else {
        el.classList.add('hidden');
    }
}

function renderAllViews() {
    if (!appState.prediction || !appState.propertyInput) {
        renderEmptyStates();
        return;
    }

    // Toggle Empty States off
    toggleVisibility('result-empty-state', false);
    toggleVisibility('result-content-view', true);

    toggleVisibility('explain-empty-state', false);
    toggleVisibility('explain-content-view', true);

    toggleVisibility('market-empty-state', false);
    toggleVisibility('market-content-view', true);

    toggleVisibility('insights-empty-state', false);
    toggleVisibility('insights-content-view', true);

    // Render Each Section
    renderOverviewRecentValuation(appState.prediction);
    renderPredictionResult();
    renderExplainableAI();
    renderMarketIntelligence();
    renderAIInsights();
}

// ----------------------------------------------------
// 1. OVERVIEW RENDERING
// ----------------------------------------------------
function renderOverviewRecentValuation(pred) {
    const container = document.getElementById('recent-valuation-container');
    const badge = document.getElementById('recent-val-badge');
    const snapPos = document.getElementById('snap-curr-pos');
    if (!container) return;

    if (!pred || !appState.propertyInput) {
        container.innerHTML = `
            <div class="recent-val-init-card">
                <div class="recent-val-stat-row">
                    <div>
                        <span class="recent-val-sublabel">Latest Valuation</span>
                        <div class="recent-val-number" id="overview-recent-val-num">₹ 0.00 Lakhs</div>
                    </div>
                    <div style="text-align: right;">
                        <span class="recent-val-sublabel">Specification Score</span>
                        <div class="recent-val-score" id="overview-recent-val-score">0/100</div>
                    </div>
                </div>
                <div class="recent-val-empty-msg" id="overview-recent-val-desc">
                    No property valuation has been computed yet in this session. Configure and submit property parameters to run the trained model pipeline.
                </div>
                <button class="btn btn-primary btn-sm" onclick="navigateTo('page-valuation')" style="margin-top: 14px;">⚡ Run Property Valuation</button>
            </div>
        `;
        if (badge) badge.textContent = 'None';
        if (snapPos) snapPos.textContent = 'No active valuation';
        return;
    }

    const inp = appState.propertyInput;
    if (badge) badge.textContent = `${inp.City} • ${inp.Property_Type}`;

    let posText = `${inp.City} ${inp.Locality_Type} ${inp.Property_Type}`;
    if (appState.marketComparison) {
        posText += ` — ${appState.marketComparison.city_percentile}th Percentile (${appState.marketComparison.market_position_tier})`;
    }
    if (snapPos) snapPos.textContent = posText;

    container.innerHTML = `
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 18px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px;">
                <div>
                    <span style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #64748B;">Estimated Market Value</span>
                    <div style="font-size: 1.85rem; font-weight: 900; color: #0F172A; line-height: 1.1;">₹ ${pred.Predicted_Price_Lakhs.toFixed(2)} Lakhs</div>
                    <div style="font-size: 0.85rem; color: #64748B;">≈ ₹ ${pred.Predicted_Price_Crores.toFixed(3)} Crores</div>
                </div>
                <div style="text-align: right;">
                    <span class="metric-pill" style="font-size: 0.78rem;">${pred.Valuation_Tier}</span>
                    <div style="font-size: 0.85rem; font-weight: 800; color: #2563EB; margin-top: 4px;">Score: ${pred.Valuation_Score.toFixed(1)}/100</div>
                </div>
            </div>

            <div style="font-size: 0.82rem; color: #475569; padding: 8px 12px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; margin-bottom: 14px;">
                <strong>${inp.City}</strong> (${inp.Locality_Type}) • <strong>${inp.Super_Area_SqFt.toLocaleString()} SqFt</strong> Super • ${inp.BHK} BHK / ${inp.Bathrooms} Bath • ${inp.Age_of_Property} yrs old
            </div>

            <div style="display: flex; gap: 10px;">
                <button class="btn btn-primary" onclick="navigateTo('page-result')">🎯 View Full Valuation</button>
                <button class="btn btn-secondary" onclick="navigateTo('page-explain')">🔍 Inspect SHAP Factors</button>
            </div>
        </div>
    `;
}

// ----------------------------------------------------
// 2. PREDICTION RESULT RENDERING
// ----------------------------------------------------
function renderPredictionResult() {
    const pred = appState.prediction;
    const inp = appState.propertyInput;
    if (!pred || !inp) return;

    document.getElementById('res-price-lakhs').textContent = `₹ ${pred.Predicted_Price_Lakhs.toFixed(2)} Lakhs`;
    document.getElementById('res-price-crores').textContent = `≈ ₹ ${pred.Predicted_Price_Crores.toFixed(3)} Crores`;
    document.getElementById('res-confidence-range').textContent = `Valuation Range (84% Confidence): ₹ ${pred.Valuation_Lower_Lakhs.toFixed(2)} L – ₹ ${pred.Valuation_Upper_Lakhs.toFixed(2)} L`;

    document.getElementById('res-sqft-super').textContent = `₹ ${pred.Price_Per_SqFt_Super.toLocaleString('en-IN')}`;
    document.getElementById('res-sqft-carpet').textContent = `₹ ${pred.Price_Per_SqFt_Carpet.toLocaleString('en-IN')}`;
    document.getElementById('res-score').textContent = `${pred.Valuation_Score.toFixed(1)} / 100`;
    document.getElementById('res-tier').textContent = pred.Valuation_Tier;

    // Snapshot specs
    document.getElementById('snap-city').textContent = inp.City;
    document.getElementById('snap-locality').textContent = `${inp.Locality_Type} Tier`;
    document.getElementById('snap-type').textContent = inp.Property_Type;
    document.getElementById('snap-layout').textContent = `${inp.BHK} BHK / ${inp.Bathrooms} Bath`;
    document.getElementById('snap-super').textContent = `${inp.Super_Area_SqFt.toLocaleString()} SqFt`;
    document.getElementById('snap-carpet').textContent = `${inp.Carpet_Area_SqFt.toLocaleString()} SqFt (${((inp.Carpet_Area_SqFt/inp.Super_Area_SqFt)*100).toFixed(0)}% usable)`;
    document.getElementById('snap-floor').textContent = `Floor ${inp.Floor_Number} of ${inp.Total_Floors}`;
    document.getElementById('snap-age').textContent = `${inp.Age_of_Property} Years Old`;

    document.getElementById('res-val-summary').textContent = pred.Executive_Summary;
}

// ----------------------------------------------------
// 3. EXPLAINABLE AI RENDERING
// ----------------------------------------------------
function renderExplainableAI() {
    const exp = appState.explainability;
    if (!exp) return;

    if (exp.base_value_lakhs !== null && exp.predicted_price_lakhs !== null) {
        document.getElementById('exp-base-val').textContent = `₹ ${exp.base_value_lakhs.toFixed(2)} Lakhs`;
        const sign = exp.net_impact_lakhs >= 0 ? '+' : '';
        document.getElementById('exp-net-impact').textContent = `${sign}₹ ${exp.net_impact_lakhs.toFixed(2)} Lakhs`;
        document.getElementById('exp-pred-val').textContent = `₹ ${exp.predicted_price_lakhs.toFixed(2)} Lakhs`;
    }

    // Positive Drivers
    const posList = document.getElementById('exp-positive-list');
    if (posList) {
        posList.innerHTML = '';
        if (exp.positive_drivers && exp.positive_drivers.length > 0) {
            exp.positive_drivers.forEach(driver => {
                const item = document.createElement('div');
                item.className = 'driver-item-card pos';
                item.innerHTML = `
                    <div>
                        <span class="driver-item-name">${driver.feature}</span>
                        <span class="driver-item-raw">(${driver.raw_value})</span>
                    </div>
                    <span class="driver-item-val pos">+₹ ${driver.contribution_lakhs.toFixed(2)} L</span>
                `;
                posList.appendChild(item);
            });
        } else {
            posList.innerHTML = `<div style="font-size: 0.85rem; color: #64748B; padding: 8px;">Baseline feature attributes aligned with standard dataset norms.</div>`;
        }
    }

    // Negative Drivers
    const negList = document.getElementById('exp-negative-list');
    if (negList) {
        negList.innerHTML = '';
        if (exp.negative_drivers && exp.negative_drivers.length > 0) {
            exp.negative_drivers.forEach(driver => {
                const item = document.createElement('div');
                item.className = 'driver-item-card neg';
                item.innerHTML = `
                    <div>
                        <span class="driver-item-name">${driver.feature}</span>
                        <span class="driver-item-raw">(${driver.raw_value})</span>
                    </div>
                    <span class="driver-item-val neg">-₹ ${Math.abs(driver.contribution_lakhs).toFixed(2)} L</span>
                `;
                negList.appendChild(item);
            });
        } else {
            negList.innerHTML = `<div style="font-size: 0.85rem; color: #64748B; padding: 8px;">No material negative structural discounts identified for this property.</div>`;
        }
    }

    // Horizontal Waterfall Visualization
    renderWaterfallChart(exp);

    // AI Narrative
    const narrativeEl = document.getElementById('exp-ai-narrative');
    if (narrativeEl && exp.ai_explanation) {
        narrativeEl.textContent = exp.ai_explanation;
    }

    // Reset What-If Simulator with subject values
    if (appState.propertyInput) {
        syncWhatIfInputs(appState.propertyInput, exp.predicted_price_lakhs);
    }
}

function renderWaterfallChart(exp) {
    const container = document.getElementById('waterfall-chart-container');
    if (!container || !exp.all_contributions) return;

    container.innerHTML = '';

    // Find max contribution for scale
    const contributions = exp.all_contributions;
    const maxVal = Math.max(
        exp.base_value_lakhs || 140,
        ...contributions.map(c => Math.abs(c.contribution_lakhs))
    );

    // 1. Base Value Row
    const baseRow = document.createElement('div');
    baseRow.className = 'waterfall-row';
    const basePct = Math.min(100, ((exp.base_value_lakhs || 142.5) / maxVal) * 80);
    baseRow.innerHTML = `
        <div class="waterfall-label">
            <span class="wf-title">Baseline Intercept</span>
            <span class="wf-sub">Dataset Average Price</span>
        </div>
        <div class="waterfall-bar-track">
            <div class="waterfall-bar-fill base" style="width: ${basePct}%;"></div>
        </div>
        <div class="waterfall-val base">₹ ${(exp.base_value_lakhs || 142.51).toFixed(2)} L</div>
    `;
    container.appendChild(baseRow);

    // 2. Individual Feature Contributions
    contributions.forEach(c => {
        if (Math.abs(c.contribution_lakhs) < 0.05) return; // filter trivial noise

        const row = document.createElement('div');
        row.className = 'waterfall-row';
        const widthPct = Math.min(100, (Math.abs(c.contribution_lakhs) / maxVal) * 80);
        const isPos = c.is_positive;
        const sign = isPos ? '+' : '-';
        const barClass = isPos ? 'pos' : 'neg';

        row.innerHTML = `
            <div class="waterfall-label">
                <span class="wf-title">${c.feature}</span>
                <span class="wf-sub">${c.raw_value}</span>
            </div>
            <div class="waterfall-bar-track">
                <div class="waterfall-bar-fill ${barClass}" style="width: ${widthPct}%;"></div>
            </div>
            <div class="waterfall-val ${barClass}">${sign}₹ ${Math.abs(c.contribution_lakhs).toFixed(2)} L</div>
        `;
        container.appendChild(row);
    });

    // 3. Final Prediction Row
    const finalRow = document.createElement('div');
    finalRow.className = 'waterfall-row';
    finalRow.style.borderTop = '2px solid #E2E8F0';
    finalRow.style.paddingTop = '10px';
    finalRow.style.marginTop = '6px';
    const finalPct = Math.min(100, ((exp.predicted_price_lakhs || 200) / maxVal) * 80);

    finalRow.innerHTML = `
        <div class="waterfall-label">
            <span class="wf-title" style="font-weight: 800; color: #0F172A;">FINAL VALUATION</span>
            <span class="wf-sub">Combined Machine Learning Estimate</span>
        </div>
        <div class="waterfall-bar-track">
            <div class="waterfall-bar-fill base" style="width: ${finalPct}%;"></div>
        </div>
        <div class="waterfall-val base" style="font-size: 1rem; font-weight: 900;">₹ ${(exp.predicted_price_lakhs || 0).toFixed(2)} L</div>
    `;
    container.appendChild(finalRow);
}

// Global SHAP & Feature Importances
async function loadGlobalExplainability() {
    try {
        const response = await fetch('/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        if (!response.ok) return;
        const data = await response.json();
        appState.globalExplainability = data;

        renderGlobalFeaturesList(data.feature_importances);
    } catch (err) {
        console.warn('Global explainability warning:', err);
    }
}

function renderGlobalFeaturesList(features) {
    const table = document.getElementById('top-features-table');
    if (!table || !features) return;

    table.innerHTML = '';
    features.slice(0, 10).forEach(feat => {
        const row = document.createElement('div');
        row.className = 'global-feat-row';
        row.innerHTML = `
            <div>
                <span class="global-feat-name">${feat.feature}</span>
                <p class="global-feat-desc">${feat.description || 'Model feature weight'}</p>
            </div>
            <span class="global-feat-pct">${(feat.importance * 100).toFixed(2)}%</span>
        `;
        table.appendChild(row);
    });
}

// ----------------------------------------------------
// 4. WHAT-IF PROPERTY SIMULATOR
// ----------------------------------------------------
function initWhatIfControls() {
    const bindSlider = (sliderId, labelId, suffix = '') => {
        const slider = document.getElementById(sliderId);
        const label = document.getElementById(labelId);
        if (slider && label) {
            slider.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                label.textContent = suffix === ' SqFt' ? val.toLocaleString() : val.toFixed(1);
            });
        }
    };

    bindSlider('wi-super-area', 'wi-super-val', ' SqFt');
    bindSlider('wi-dist-city', 'wi-dist-city-val', ' km');
    bindSlider('wi-dist-metro', 'wi-dist-metro-val', ' km');
    bindSlider('wi-age', 'wi-age-val', ' yrs');
}

function syncWhatIfInputs(inp, basePrice) {
    const setSlider = (sliderId, labelId, val, suffix = '') => {
        const slider = document.getElementById(sliderId);
        const label = document.getElementById(labelId);
        if (slider) slider.value = val;
        if (label) label.textContent = suffix === ' SqFt' ? val.toLocaleString() : val.toFixed(1);
    };

    setSlider('wi-super-area', 'wi-super-val', inp.Super_Area_SqFt, ' SqFt');
    setSlider('wi-dist-city', 'wi-dist-city-val', inp.Distance_to_City_Center_km, ' km');
    setSlider('wi-dist-metro', 'wi-dist-metro-val', inp.Distance_to_Metro_km, ' km');
    setSlider('wi-age', 'wi-age-val', inp.Age_of_Property, ' yrs');

    const baseEl = document.getElementById('wi-base-price');
    if (baseEl) baseEl.textContent = `₹ ${(basePrice || inp.Super_Area_SqFt * 0.1).toFixed(2)} L`;

    const simEl = document.getElementById('wi-sim-price');
    if (simEl) simEl.textContent = `₹ ${(basePrice || 0).toFixed(2)} L`;

    const diffEl = document.getElementById('wi-diff-val');
    if (diffEl) diffEl.textContent = '± ₹ 0.00 L (0.0%)';

    const sumEl = document.getElementById('wi-summary-text');
    if (sumEl) sumEl.textContent = 'Adjust parameters on the left and click "Simulate Value" to compute real model inference.';
}

async function runWhatIfSimulation() {
    if (!appState.propertyInput) {
        showToast('⚠️ Please run a property valuation first.');
        return;
    }

    const btn = document.getElementById('wi-simulate-btn');
    btn.textContent = '⏳ Calculating...';
    btn.disabled = true;

    try {
        const baseProp = appState.propertyInput;
        const newSuper = parseFloat(document.getElementById('wi-super-area').value);
        const newDistCity = parseFloat(document.getElementById('wi-dist-city').value);
        const newDistMetro = parseFloat(document.getElementById('wi-dist-metro').value);
        const newAge = parseInt(document.getElementById('wi-age').value, 10);

        // Keep realistic carpet ratio
        const ratio = baseProp.Carpet_Area_SqFt / baseProp.Super_Area_SqFt;
        const newCarpet = Math.round(newSuper * ratio);

        const modProp = {
            ...baseProp,
            Super_Area_SqFt: newSuper,
            Carpet_Area_SqFt: newCarpet,
            Distance_to_City_Center_km: newDistCity,
            Distance_to_Metro_km: newDistMetro,
            Age_of_Property: newAge
        };

        const response = await fetch('/what-if', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                base_property: baseProp,
                modified_property: modProp
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'What-if calculation failed');
        }

        const data = await response.json();
        document.getElementById('wi-base-price').textContent = `₹ ${data.base_price_lakhs.toFixed(2)} L`;
        document.getElementById('wi-sim-price').textContent = `₹ ${data.simulated_price_lakhs.toFixed(2)} L`;

        const diffEl = document.getElementById('wi-diff-val');
        const sign = data.difference_lakhs >= 0 ? '+' : '';
        diffEl.textContent = `${sign}₹ ${Math.abs(data.difference_lakhs).toFixed(2)} L (${sign}${data.difference_percentage.toFixed(1)}%)`;
        diffEl.style.color = data.difference_lakhs >= 0 ? '#059669' : '#DC2626';

        document.getElementById('wi-summary-text').textContent = data.summary;

        showToast('✅ Scenario simulation completed.');

    } catch (err) {
        showToast(`Simulation Error: ${err.message}`);
        console.error('What-if error:', err);
    } finally {
        btn.textContent = '⚡ SIMULATE VALUE';
        btn.disabled = false;
    }
}

// ----------------------------------------------------
// 5. MARKET INTELLIGENCE RENDERING
// ----------------------------------------------------
function renderMarketIntelligence() {
    const mkt = appState.marketComparison;
    if (!mkt) return;

    document.getElementById('mkt-prop-val').textContent = `₹ ${mkt.property_price_lakhs.toFixed(2)} L`;
    document.getElementById('mkt-city-med').textContent = `₹ ${mkt.city_median_lakhs.toFixed(2)} L`;
    document.getElementById('mkt-city-name').textContent = `${mkt.city} Market Median`;
    document.getElementById('mkt-sqft').textContent = `₹ ${mkt.price_per_sqft.toLocaleString('en-IN')}`;

    const diffSign = mkt.market_diff_percentage >= 0 ? '+' : '';
    const diffEl = document.getElementById('mkt-diff');
    diffEl.textContent = `${diffSign}${mkt.market_diff_percentage.toFixed(1)}%`;
    diffEl.style.color = mkt.market_diff_percentage >= 0 ? '#059669' : '#DC2626';

    document.getElementById('mkt-pctile').textContent = `${mkt.city_percentile.toFixed(1)}th`;
    document.getElementById('mkt-nat-pctile').textContent = `${mkt.national_percentile.toFixed(1)}th National`;

    // Position Description & Scale
    document.getElementById('mkt-position-desc').textContent = mkt.position_description;

    const marker = document.getElementById('mkt-position-marker');
    const markerLabel = document.getElementById('mkt-marker-label');
    if (marker && markerLabel) {
        const clampedPct = Math.min(98, Math.max(2, mkt.city_percentile));
        marker.style.left = `${clampedPct}%`;
        markerLabel.textContent = `Your Property: ${mkt.city_percentile.toFixed(1)}th Percentile (${mkt.market_position_tier})`;
    }

    // Comparable Properties Table
    renderComparablesTable(mkt.comparable_properties, mkt);

    // Price vs Area Insight
    document.getElementById('mkt-price-area-insight').textContent = mkt.price_vs_area_insight;

    // City Market Comparison Table
    renderCityComparisonTable(mkt.city_comparison);
}

function renderComparablesTable(comps, mkt) {
    const tbody = document.getElementById('comps-table-body');
    if (!tbody) return;

    tbody.innerHTML = '';

    // 1. Top Highlight Row for Subject Property
    const subjectRow = document.createElement('tr');
    subjectRow.className = 'row-selected';
    subjectRow.innerHTML = `
        <td><strong>SUBJECT PROPERTY</strong></td>
        <td>${mkt.city} (${appState.propertyInput.Locality_Type})</td>
        <td>${appState.propertyInput.Property_Type} • ${appState.propertyInput.BHK} BHK</td>
        <td>${appState.propertyInput.Super_Area_SqFt.toLocaleString()} SqFt</td>
        <td>${appState.propertyInput.Age_of_Property} yrs</td>
        <td><strong>₹ ${mkt.property_price_lakhs.toFixed(2)} L (AI)</strong></td>
        <td>₹ ${mkt.price_per_sqft.toLocaleString('en-IN')}</td>
        <td><span class="badge badge-blue">Baseline Target</span></td>
    `;
    tbody.appendChild(subjectRow);

    // 2. Verified Comparable Historical Transactions
    if (comps && comps.length > 0) {
        comps.forEach(comp => {
            const tr = document.createElement('tr');
            const diffSign = comp.price_diff_vs_subject_lakhs >= 0 ? '+' : '';
            const diffColor = comp.price_diff_vs_subject_lakhs >= 0 ? '#059669' : '#DC2626';

            tr.innerHTML = `
                <td><code>${comp.property_id}</code></td>
                <td>${comp.city} (${comp.locality_type})</td>
                <td>${comp.property_type} • ${comp.bhk} BHK</td>
                <td>${comp.super_area_sqft.toLocaleString()} SqFt</td>
                <td>${comp.age_of_property} yrs</td>
                <td>₹ ${comp.price_lakhs.toFixed(2)} L</td>
                <td>₹ ${comp.price_per_sqft.toLocaleString('en-IN')}</td>
                <td style="color: ${diffColor}; font-weight: 700;">${diffSign}₹ ${comp.price_diff_vs_subject_lakhs.toFixed(2)} L</td>
            `;
            tbody.appendChild(tr);
        });
    } else {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="8" style="text-align: center; color: #64748B;">No direct comparable transactions available in this exact criteria bracket.</td>`;
        tbody.appendChild(tr);
    }
}

function renderCityComparisonTable(cityStats) {
    const tbody = document.getElementById('city-market-table-body');
    if (!tbody || !cityStats) return;

    tbody.innerHTML = '';
    cityStats.forEach(stat => {
        const tr = document.createElement('tr');
        if (stat.is_selected) {
            tr.className = 'row-selected';
        }

        const badgeHtml = stat.is_selected 
            ? `<span class="badge badge-blue">Evaluated Metro</span>` 
            : `<span class="badge badge-neutral">Metro Registry</span>`;

        tr.innerHTML = `
            <td><strong>${stat.city}</strong></td>
            <td>₹ ${stat.median_price_lakhs.toFixed(2)} L</td>
            <td>₹ ${stat.mean_price_lakhs.toFixed(2)} L</td>
            <td>₹ ${stat.median_price_per_sqft.toLocaleString('en-IN')}</td>
            <td>${stat.count.toLocaleString()}</td>
            <td>${badgeHtml}</td>
        `;
        tbody.appendChild(tr);
    });
}

// ----------------------------------------------------
// 6. AI INSIGHTS RENDERING
// ----------------------------------------------------
function renderAIInsights() {
    const pred = appState.prediction;
    const mkt = appState.marketComparison;
    const inp = appState.propertyInput;
    if (!pred || !inp) return;

    document.getElementById('ins-tier-title').textContent = pred.Valuation_Tier.toUpperCase();
    document.getElementById('ins-score-num').textContent = pred.Valuation_Score.toFixed(1);

    if (mkt) {
        document.getElementById('ins-tier-sub').textContent = `${mkt.market_position_tier} (${mkt.city_percentile.toFixed(1)}th Percentile in ${inp.City})`;
    }

    // Key Drivers List
    const driversList = document.getElementById('ins-drivers-list');
    if (driversList) {
        driversList.innerHTML = '';
        if (pred.Key_Value_Drivers && pred.Key_Value_Drivers.length > 0) {
            pred.Key_Value_Drivers.forEach(d => {
                const li = document.createElement('li');
                li.textContent = d;
                driversList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'Property aligns with standard median market parameters.';
            driversList.appendChild(li);
        }
    }

    // Discounts List
    const discountsList = document.getElementById('ins-discounts-list');
    if (discountsList) {
        discountsList.innerHTML = '';
        if (pred.Discount_Factors && pred.Discount_Factors.length > 0) {
            pred.Discount_Factors.forEach(d => {
                const li = document.createElement('li');
                li.textContent = d;
                discountsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No significant structural discount factors identified by the model.';
            discountsList.appendChild(li);
        }
    }

    // Positioning
    const posEl = document.getElementById('ins-market-pos-text');
    if (posEl) {
        if (mkt) {
            posEl.textContent = mkt.position_description;
        } else {
            posEl.textContent = `Property is evaluated at ₹ ${pred.Predicted_Price_Lakhs.toFixed(2)} Lakhs under the trained model pipeline.`;
        }
    }

    // Summary
    const sumEl = document.getElementById('ins-ai-summary');
    if (sumEl) {
        sumEl.textContent = pred.Executive_Summary;
    }

    // Decision Support
    renderDecisionSupport(pred, mkt, inp);
}

function renderDecisionSupport(pred, mkt, inp) {
    const sellerEl = document.getElementById('ins-seller-guidance');
    const buyerEl = document.getElementById('ins-buyer-guidance');

    if (sellerEl) {
        const lowerL = pred.Valuation_Lower_Lakhs.toFixed(2);
        const upperL = pred.Valuation_Upper_Lakhs.toFixed(2);
        sellerEl.innerHTML = `
            <p><strong>Recommended Indicative Listing Range:</strong> ₹ ${lowerL} L to ₹ ${upperL} Lakhs.</p>
            <p style="margin-top: 6px;">
                Positioning this listing within this statistical corridor preserves competitive exposure while capitalizing on the property's ${inp.Locality_Type} locality premium and modern vintage (${inp.Age_of_Property} years). Setting an asking price above ₹ ${upperL} L may extend market listing duration.
            </p>
        `;
    }

    if (buyerEl) {
        const unitSuper = pred.Price_Per_SqFt_Super.toLocaleString('en-IN');
        const unitCityMed = mkt ? `₹ ${mkt.city_median_price_per_sqft.toLocaleString('en-IN')}` : 'city median';
        buyerEl.innerHTML = `
            <p><strong>Fair Acquisition Benchmark:</strong> ₹ ${pred.Predicted_Price_Lakhs.toFixed(2)} Lakhs (₹ ${unitSuper}/SqFt).</p>
            <p style="margin-top: 6px;">
                The valuation score of <strong>${pred.Valuation_Score.toFixed(1)}/100</strong> classifies this unit as a <em>${pred.Valuation_Tier}</em>. If negotiating against an asking quote, compare against the local ${inp.City} unit median (${unitCityMed}/SqFt) and verify whether amenities match the recorded specifications.
            </p>
        `;
    }
}

// ----------------------------------------------------
// TOAST HELPER
// ----------------------------------------------------
function showToast(message) {
    const toast = document.getElementById('toast');
    if (!toast) return;

    toast.textContent = message;
    toast.classList.remove('hidden');

    setTimeout(() => {
        toast.classList.add('hidden');
    }, 4500);
}
