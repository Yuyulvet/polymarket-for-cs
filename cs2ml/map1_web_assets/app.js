"use strict";
let state = null, selectedId = null, csrf = null, refreshing = false, actionPending = false;
let hltvEvidenceId = null, hltvIdentitySignature = null;
const $ = id => document.getElementById(id);
const form = $("context-form");
const currency = v => Number(v).toLocaleString("en-US", {style:"currency", currency:"USD"});
const timestamp = v => v ? new Date(v).toLocaleString("zh-CN", {hour12:false}) : "—";
const localInput = v => { const d = v ? new Date(v) : new Date(); return new Date(d.getTime()-d.getTimezoneOffset()*60000).toISOString().slice(0,19); };
const reasonNames = {map_and_roster_unconfirmed:"地图与阵容尚未确认",prematch_window_closed:"赛前窗口已关闭",market_closed:"市场已关闭",insufficient_roster_history:"阵容历史不足",edge_below_threshold:"扣费后优势未达门槛",no_confirmed_events_in_capture_window:"窗口内没有已确认比赛",signal_no_longer_valid:"信号已失效",missing_timely_post_delay_book:"延迟后未及时收到新盘口",book_predates_eligibility:"盘口早于允许成交时刻",insufficient_depth_at_limit:"限价内深度不足",another_operation_is_running:"上一项操作仍在执行",context_locked_after_first_observation:"已经采集，不能再修改历史确认信息"};
Object.assign(reasonNames, {
  invalid_individual_steamid64:"请输入有效的个人 SteamID64",
  hltv_import_requires_open_prematch_event:"仅可导入尚未开赛且开放的比赛",
  hltv_rate_limit_wait_15_seconds:"HLTV 读取间隔至少 15 秒，请稍后重试",
  hltv_request_cooldown:"HLTV 读取冷却中",
  hltv_source_unavailable_manual_retry_required:"HLTV 读取失败，已停止自动重试；请人工检查后重试",
  hltv_evidence_stale_or_future:"HLTV 证据已过期或时间异常，请重新读取",
  hltv_match_not_prematch:"HLTV 比赛不是可确认的赛前状态",
  hltv_map1_not_announced:"HLTV 尚未公布第一张地图",
  hltv_market_not_prematch:"对应市场已不在可用的赛前状态",
  hltv_team_identity_mismatch:"HLTV 双方与市场队伍不一致",
  hltv_schedule_mismatch:"HLTV 与市场开赛时间相差超过允许范围",
  hltv_missing_match_identity_or_schedule:"缺少可靠的比赛身份或赛程信息",
  hltv_incomplete_lineup:"HLTV 本场五人名单不完整",
  hltv_incomplete_or_duplicate_players:"本场选手名单不完整或包含重复选手",
  hltv_player_identities_unverified:"仍有选手未核实 HLTV 与 SteamID 对应关系",
  hltv_overlapping_steam_ids:"本场不同选手对应了相同 SteamID",
  hltv_match_already_bound_to_another_event:"此 HLTV 比赛已绑定其他市场比赛",
  hltv_evidence_event_mismatch:"此证据不属于当前选择的比赛",
  hltv_evidence_superseded_reload_review:"已有更新的 HLTV 证据，请重新核对",
  hltv_identity_attestation_required:"请先勾选身份核实声明",
  hltv_valid_source_required:"需要成功读取的 HLTV 来源才能保存身份",
  hltv_invalid_identity_batch:"每次需提交 1 至 10 位已核实选手",
  hltv_invalid_identity_or_missing_evidence:"选手身份无效、重复或缺少核实依据",
  hltv_match_review_required:"请先勾选比赛复核声明",
  hltv_source_evidence_missing:"缺少 HLTV 原始来源证据",
  hltv_market_evidence_missing:"缺少对应市场的来源证据",
  hltv_map_roster_or_match_changed_requires_new_review:"地图、名单或比赛信息已变更，需重新复核；已采集记录不会改写",
  hltv_evidence_not_found:"未找到指定 HLTV 证据",
  hltv_identity_conflict_requires_review:"身份映射与已保存记录冲突，需要人工检查",
  manual_context_cannot_claim_hltv_verification:"人工备用确认不能冒充 HLTV 自动核实",
  local_identity_hints_unavailable:"本地选手候选不可用",
  html_missing_or_oversized:"页面为空或超过大小限制",
  access_challenge:"HLTV 返回访问验证页面；不会绕过限制",
  html_structure_invalid:"页面结构无法安全解析",
  match_header_missing_or_ambiguous:"页面缺少唯一的比赛信息区",
  canonical_match_id_mismatch:"页面声明的比赛编号与链接不一致",
  canonical_match_url_invalid:"页面声明的比赛链接无效",
  scheduled_start_invalid:"页面开赛时间格式无效",
  scheduled_start_missing_or_ambiguous:"页面缺少唯一的开赛时间",
  event_name_missing_or_ambiguous:"页面缺少明确的赛事名称",
  time_and_event_missing_or_ambiguous:"页面赛程与赛事信息不完整或有歧义",
  match_status_unknown:"无法判断比赛是否尚未开赛",
  team_header_missing_or_ambiguous:"队伍信息区缺失或有歧义",
  team_identity_missing_or_ambiguous:"队伍身份缺失或有歧义",
  requires_two_distinct_teams:"必须识别出两个不同的队伍",
  map_name_unknown:"无法识别地图名称，地图序号",
  map_holders_missing:"页面未提供地图顺序记录",
  maps_section_missing_or_ambiguous:"页面地图区域缺失或有歧义",
  map1_not_announced:"第一张地图尚未公布（TBA）",
  lineups_section_missing_or_ambiguous:"页面本场名单区域缺失或有歧义",
  lineup_team_identity_missing_or_ambiguous:"名单所属队伍不明确",
  lineup_header_team_mismatch:"名单队伍与比赛双方不一致",
  lineup_missing_or_ambiguous:"本场队伍名单缺失或有歧义，队伍编号",
  lineup_players_section_missing_or_ambiguous:"选手列表缺失或有歧义，队伍编号",
  ambiguous_player_cell:"选手信息存在歧义，队伍编号",
  missing_or_ambiguous_player_identity:"选手身份缺失或有歧义，队伍编号",
  invalid_player_identity:"选手身份无效，队伍编号",
  lineup_requires_five_players:"本场名单必须正好有五位选手，队伍编号",
  duplicate_lineup_player:"队伍名单存在重复选手，队伍编号",
  players_not_distinct_across_teams:"双方名单存在重复选手"
});
function reason(value) {
  if(!value) return "—";
  const raw = String(value);
  if(reasonNames[raw]) return reasonNames[raw];
  const wrapper = raw.match(/^([A-Za-z][\w.]*Error):\s*(.+)$/);
  if(wrapper) { const translated = reason(wrapper[2]); return translated === wrapper[2] ? raw : `${wrapper[1]}：${translated}`; }
  if(raw.includes(",")) {
    const parts = raw.split(",").map(part => part.trim()), translated = parts.map(reason);
    if(parts.some((part,index) => part !== translated[index])) return translated.join("；");
  }
  const split = raw.indexOf(":");
  if(split > 0 && reasonNames[raw.slice(0,split)]) return `${reasonNames[raw.slice(0,split)]}：${raw.slice(split+1).trim()}`;
  return raw;
}
function element(tag, text, className) { const e = document.createElement(tag); if(text !== undefined) e.textContent = text; if(className) e.className = className; return e; }
function error(text) { $("error").hidden = !text; $("error").textContent = text || ""; }
async function action(name, body={}) {
  if(actionPending) return;
  actionPending = true;
  error("");
  if(state) render();
  try {
    const r = await fetch(`/api/${name}`, {method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrf},body:JSON.stringify(body)});
    const data = await r.json(); if(!r.ok) throw new Error(reason(data.error));
    await refresh();
  } catch(e) { error(e.message); }
  finally { actionPending = false; if(state) render(); }
}
function selectEvent(event) {
  selectedId = event.event_id;
  $("no-selection").hidden = true; $("selection").hidden = false; form.reset();
  $("hltv-import-form").reset(); $("hltv-confirm-form").reset(); $("hltv-identities-form").reset();
  $("manual-confirmation").open = false;
  hltvEvidenceId = null; hltvIdentitySignature = null;
  $("hltv-players").replaceChildren();
  $("hltv-url").value = event.hltv?.url || "";
  $("selected-title").textContent = event.title;
  $("selected-time").textContent = `比赛 #${event.event_id} · 开赛 ${timestamp(event.scheduled_start_at)}`;
  $("team-a-label").textContent = event.outcomes[0]; $("team-b-label").textContent = event.outcomes[1];
  const c = event.context;
  if(c) {
    for(const name of ["map_name","map_source","roster_source"]) form.elements[name].value = c[name];
    // Stored contexts may use either team orientation; populate by exact outcome.
    const teams = [c.team_a,c.team_b];
    form.elements.roster_a.value = teams.find(t=>t.outcome===event.outcomes[0]).roster.join("\n");
    form.elements.roster_b.value = teams.find(t=>t.outcome===event.outcomes[1]).roster.join("\n");
  }
  for(const key of ["map","roster"]) form.elements[`${key}_known_at`].value = localInput(c?.[`declared_${key}_known_at`]);
  const locked = Boolean(event.latest) || event.status === "excluded";
  for(const field of form.elements) field.disabled = locked;
  $("confirm").textContent = locked ? "已锁定：保留原始确认记录" : "保存确认（仅纸面）";
  renderEvents(); renderHltv(event);
}
function hltvFresh(evidence) {
  const age = Date.now() - new Date(evidence?.received_at).getTime();
  return Number.isFinite(age) && age >= 0 && age <= 15 * 60 * 1000;
}
function renderHltvPlayers(evidence) {
  const sameEvidence = hltvEvidenceId === evidence.evidence_id;
  const signature = JSON.stringify((evidence.teams || []).map(team => [team.hltv_team_id, team.outcome,
    (team.players || []).map(player => [player.hltv_player_id, player.steamid, player.verified, player.candidates])]));
  if(sameEvidence && signature === hltvIdentitySignature) return;
  const drafts = new Map();
  if(sameEvidence) for(const row of $("hltv-players").querySelectorAll(".identity-edit")) {
    drafts.set(row.dataset.playerId, {steamid:row.querySelector(".identity-steamid").value, source:row.querySelector(".identity-source").value});
  }
  hltvEvidenceId = evidence.evidence_id; hltvIdentitySignature = signature;
  $("hltv-reviewed").checked = false; $("hltv-identities-attest").checked = false;
  const root = $("hltv-players"); root.replaceChildren();
  for(const team of evidence.teams || []) {
    const block = element("section", undefined, "hltv-team");
    block.append(element("h4",team.name || "未识别队伍"));
    block.append(element("p",team.outcome ? `对应市场选项：${team.outcome}` : "尚未匹配市场队伍", "hint"));
    for(const player of team.players || []) {
      const verified = Boolean(player.verified && player.steamid);
      const row = element("div", undefined, "hltv-player" + (verified ? "" : " identity-edit"));
      const playerId = String(player.hltv_player_id ?? "");
      row.dataset.playerId = playerId;
      const heading = element("div", undefined, "hltv-heading");
      heading.append(element("strong",player.name || "未识别选手"),element("span",`HLTV #${playerId}`,"muted"));
      row.append(heading);
      if(verified) row.append(element("p",`已核实 SteamID · ${player.steamid}`,"identity-verified"));
      else {
        const candidates = player.candidates || [];
        let select = null;
        if(candidates.length) {
          const label = element("label", "本地历史候选（须自行核实）");
          select = element("select", undefined, "identity-candidate");
          select.append(new Option("请选择，或在下方手动填写", ""));
          candidates.forEach((candidate,index) => select.append(new Option(`${candidate.name || "历史选手"} · ${candidate.steamid}`, String(index))));
          label.append(select); row.append(label);
        } else row.append(element("p","未找到可直接确认的历史候选，请核实身份后填写。","hint"));
        const idLabel = element("label", "SteamID64");
        const idInput = element("input", undefined, "identity-steamid");
        idInput.type = "text"; idInput.inputMode = "numeric"; idInput.maxLength = 17;
        idInput.placeholder = "17 位 SteamID64"; idInput.autocomplete = "off";
        const sourceLabel = element("label", "身份核实依据");
        const sourceInput = element("input", undefined, "identity-source");
        sourceInput.type = "text"; sourceInput.maxLength = 1000;
        sourceInput.placeholder = "选手主页链接 + 本地 demo / 其他身份依据";
        const draft = drafts.get(playerId);
        if(draft) { idInput.value = draft.steamid; sourceInput.value = draft.source; }
        idLabel.append(idInput); sourceLabel.append(sourceInput); row.append(idLabel, sourceLabel);
        if(select) {
          select.addEventListener("change", () => {
            if(select.value === "") return;
            const candidate = candidates[Number(select.value)];
            idInput.value = String(candidate.steamid || ""); sourceInput.value = String(candidate.source || "");
            $("hltv-identities-attest").checked = false; syncHltvControls();
          });
        }
        for(const input of [idInput,sourceInput]) input.addEventListener("input", () => {
          $("hltv-identities-attest").checked = false; syncHltvControls();
        });
      }
      block.append(row);
    }
    root.append(block);
  }
}
function renderHltv(event) {
  const evidence = event?.hltv;
  $("hltv-empty").hidden = Boolean(evidence); $("hltv-evidence").hidden = !evidence;
  if(evidence) {
    if(!$("hltv-url").value) $("hltv-url").value = evidence.url || "";
    const ready = evidence.ready && hltvFresh(evidence);
    $("hltv-map").textContent = evidence.map_name ? `Map 1 · ${evidence.map_name}` : "Map 1 · TBA / 未核实";
    $("hltv-status").textContent = evidence.error ? "读取失败" : !hltvFresh(evidence) ? "证据已过期" : evidence.confirmation_matches === false ? "与确认记录不一致" : ready && evidence.confirmation_matches === true ? "已确认 · 证据有效" : ready ? "待人工复核" : "信息待补全";
    $("hltv-status").className = "tag" + (ready && evidence.confirmation_matches !== false ? "" : " wait");
    $("hltv-meta").textContent = [evidence.match_id ? `HLTV #${evidence.match_id}` : null,
      evidence.event_name, `读取于 ${timestamp(evidence.received_at)}`,
      evidence.scheduled_start_at ? `页面开赛 ${timestamp(evidence.scheduled_start_at)}` : null,
      evidence.sha256 ? `证据 SHA-256 ${evidence.sha256}` : null].filter(Boolean).join(" · ");
    $("hltv-error").hidden = !evidence.error; $("hltv-error").textContent = evidence.error ? reason(evidence.error) : "";
    $("hltv-issues").replaceChildren();
    for(const issue of evidence.issues || []) $("hltv-issues").append(element("li",reason(typeof issue === "string" ? issue : JSON.stringify(issue))));
    if(evidence.candidate_warning) $("hltv-issues").append(element("li",reason(evidence.candidate_warning)));
    $("hltv-issues").hidden = !$("hltv-issues").children.length;
    renderHltvPlayers(evidence);
  }
  syncHltvControls();
}
function syncHltvControls() {
  const selected = state?.events.find(event => event.event_id === selectedId);
  const evidence = selected?.hltv;
  const busy = Boolean(state?.active_job || actionPending);
  const excluded = !selected || selected.status === "excluded";
  $("hltv-import").disabled = busy || excluded;
  $("hltv-url").disabled = busy || excluded;
  const rows = [...$("hltv-players").querySelectorAll(".identity-edit")];
  $("hltv-identities-form").hidden = !rows.length || !evidence;
  for(const field of $("hltv-players").querySelectorAll("input,select")) field.disabled = busy || excluded;
  $("hltv-identities-attest").disabled = busy || excluded;
  $("hltv-identities-save").disabled = busy || excluded || !evidence?.evidence_id || !$("hltv-identities-attest").checked;
  const locked = Boolean(selected?.latest) || excluded;
  const canConfirm = !busy && !locked && evidence?.ready && hltvFresh(evidence);
  $("hltv-reviewed").disabled = !canConfirm;
  $("hltv-confirm").disabled = !canConfirm || !$("hltv-reviewed").checked;
  $("hltv-confirm").textContent = locked ? "已锁定：保留原始确认记录" : "复核并确认比赛（仅纸面）";
}
function renderEvents() {
  const root = $("events"); root.replaceChildren();
  const entries = state.events.filter(e => $("show-excluded").checked || e.status !== "excluded");
  $("event-count").textContent = entries.length;
  if(!entries.length) { root.append(element("div",state.events.length ? "当前没有符合条件的赛前比赛。可查看排除记录，或稍后刷新赛程。" : "点击「刷新公开赛程」，开始建立观察队列。不会自动推断地图或首发。","empty")); return; }
  for(const e of entries) {
    const button = element("button",undefined,"event" + (selectedId===e.event_id?" selected":""));
    const top = element("div",undefined,"event-top");
    const stale = e.latest && Date.now()-new Date(e.latest.received_at).getTime()>15000;
    const label = e.settled ? "已核验结算" : e.settlement_check?.status==="requires_review" ? "结算待核验" : e.status==="excluded" ? "已排除" : !e.confirmed ? "待确认" : stale ? "已确认 · 快照过期" : "已确认";
    top.append(element("span",`#${e.event_id} · ${timestamp(e.scheduled_start_at)}`),element("span",label,"tag"+(!e.confirmed||stale?" wait":"")));
    button.append(top,element("h3",e.title));
    let detail = reason(e.reason);
    if(e.latest) { const d=e.latest.decision; detail = `最近采集 ${timestamp(e.latest.received_at)} · ${d.action==="signal" ? "纸面候选，等待延迟后盘口" : reason(d.reason)}`; if(e.latest.p_model!==undefined) detail+=` · 模型 A ${(e.latest.p_model*100).toFixed(1)}%`; }
    else if(e.confirmed) detail="等待采集窗口 · 地图与名单已人工确认";
    if(e.hltv) detail += ` · HLTV ${e.hltv.error ? "读取失败" : !hltvFresh(e.hltv) ? "证据过期" : e.hltv.confirmation_matches === false ? "与确认记录不一致" : e.hltv.ready && e.hltv.confirmation_matches === true ? "证据有效" : e.hltv.ready ? "证据已导入" : "待补全 / 复核"}`;
    if(e.settlement_check?.reason) detail += ` · ${e.settlement_check.reason}`;
    button.append(element("div",detail,"event-bottom"));
    button.disabled = !e.outcomes;
    button.addEventListener("click",()=>selectEvent(e)); root.append(button);
  }
}
function renderReport() {
  const r=state.report, cmp=r.probability_comparison_first_valid_snapshot_per_event;
  $("cash").textContent=currency(r.cash); $("pnl").textContent=currency(r.realized_pnl);
  $("pnl").className=r.realized_pnl<0?"negative":r.realized_pnl>0?"positive":"";
  $("open").textContent=currency(r.open_cost_basis); $("samples").textContent=`${r.observed_events} / ${r.resolved_events}`;
  $("snapshots").textContent=`${r.snapshot_count} 条快照 · ${r.pending_count} 笔待模拟成交`;
  $("validation").replaceChildren();
  const rows=[["有效二元结算观察",`${cmp.n} 场`],["模型 Brier",cmp.model?.brier?.toFixed(4)??"待积累"],["市场 Brier",cmp.market?.brier?.toFixed(4)??"待积累"],["组合共同样本",`${cmp.hybrid_common_sample?.hybrid?.n??0} 场`],["已结算纸面交易",`${r.settled_trade_count} 笔`],["实盘状态","未授权 / 未验证"]];
  for(const [name,value] of rows){const row=element("div");row.append(element("dt",name),element("dd",value));$("validation").append(row);}
  $("reasons").replaceChildren();
  const reasons=Object.entries(r.skip_reasons).sort((a,b)=>b[1]-a[1]);
  if(!reasons.length) $("reasons").append(element("p","暂无跳过或取消记录。未确认比赛不会生成预测。","muted"));
  for(const [why,count] of reasons.slice(0,12)){const row=element("div",undefined,"reason");row.append(element("span",reason(why)),element("strong",count));$("reasons").append(row);}
  $("ledger").replaceChildren();
  if(!r.ledger.length) $("ledger").append(element("p","暂无成交或跳过记录","empty"));
  const typeNames={pending:"待成交",fill:"模拟成交",settle:"结算",skip:"跳过",cancel:"取消"};
  for(const entry of [...r.ledger].reverse()){
    const row=element("div",undefined,"record");
    const detail=entry.reason?reason(entry.reason):entry.type==="settle"?`已实现 ${currency(entry.pnl)}`:entry.type==="fill"?`${entry.shares} 股 · 成本 ${currency(entry.cash)} · 费用 ${currency(entry.fee)}`:`限价 ${entry.limit}`;
    row.append(element("span",timestamp(entry.at)),element("span",`#${entry.event_id}`),element("span",typeNames[entry.type]),element("span",detail,"detail"));$("ledger").append(row);
  }
}
function render() {
  $("connection").textContent="本地服务已连接 · 只读外部接口"; $("updated").textContent=`更新于 ${timestamp(state.as_of)}`;
  $("runner").textContent=state.running?"纸面采集已启动":"采集已暂停";$("run-dot").className="dot"+(state.running?" on":"");
  $("job").textContent=state.active_job?`正在执行：${state.active_job}`:"无正在执行的任务";
  $("toggle").textContent=state.running?"暂停采集":"启动纸面采集";
  for(const id of ["discover","capture","settle"]) $(id).disabled=Boolean(state.active_job || actionPending);
  $("toggle").disabled=Boolean(state.data_error)&&!state.running;
  if(state.data_error) error("研究数据尚未准备好，请先运行 prepare，再重启工作台。"+state.data_error);
  const last=state.runs[0]; if(last?.result.status==="error") error(`最近操作失败：${reason(last.result.reason)}`);
  $("warnings").hidden=!state.warnings?.length;$("warnings").textContent=(state.warnings||[]).map(reason).join(" · ");
  renderEvents();renderReport();$("runs").replaceChildren();
  for(const run of state.runs){const row=element("div",undefined,"run");row.append(element("strong",`${timestamp(run.at)} · ${run.kind}`),element("p",JSON.stringify(run.result)));$("runs").append(row);}
  $("database").textContent=state.database;
  const selected=state.events.find(e=>e.event_id===selectedId);
  if(selected) {
    const locked = Boolean(selected.latest) || selected.status === "excluded";
    for(const field of form.elements) field.disabled = locked || Boolean(state.active_job || actionPending);
    $("confirm").textContent = locked ? "已锁定：保留原始确认记录" : "保存确认（仅纸面）";
    renderHltv(selected);
  }
}
async function refresh() {
  if(refreshing)return;refreshing=true;
  try {const r=await fetch("/api/state");const data=await r.json();if(!r.ok)throw new Error(data.error);state=data;csrf=data.csrf_token;render();}
  catch(e){$("connection").textContent="本地服务未连接";error(e.message);}
  finally{refreshing=false;}
}
for(const name of ["discover","capture","settle"]) $(name).addEventListener("click",()=>action(name));
$("toggle").addEventListener("click",()=>action(state.running?"stop":"start"));
$("show-excluded").addEventListener("change",renderEvents);
$("hltv-import-form").addEventListener("submit", async event => {
  event.preventDefault();
  if(!selectedId) return;
  const url = $("hltv-url").value.trim();
  let parsed;
  try { parsed = new URL(url); } catch { error("请输入完整的 HLTV 比赛链接。"); return; }
  if(parsed.protocol !== "https:" || parsed.hostname !== "www.hltv.org" || parsed.port || parsed.username || parsed.password || parsed.search || parsed.hash || !/^https:\/\/www\.hltv\.org\/matches\/[1-9][0-9]{0,11}(?:\/[A-Za-z0-9-]+)?\/?$/.test(url)) {
    error("只支持 https://www.hltv.org/matches/比赛编号/… 公开比赛链接。"); return;
  }
  await action("hltv_import", {event_id:selectedId,url});
});
$("hltv-identities-attest").addEventListener("change", syncHltvControls);
$("hltv-reviewed").addEventListener("change", syncHltvControls);
$("hltv-identities-form").addEventListener("submit", async event => {
  event.preventDefault();
  const selected = state?.events.find(entry => entry.event_id === selectedId);
  if(!selected?.hltv?.evidence_id || !$("hltv-identities-attest").checked) return;
  const mappings = [];
  for(const row of $("hltv-players").querySelectorAll(".identity-edit")) {
    const steamid = row.querySelector(".identity-steamid").value.trim();
    const source = row.querySelector(".identity-source").value.trim();
    if(!steamid && !source) continue;
    if(!/^\d{17}$/.test(steamid) || !source) { error("每项身份映射都需要 17 位 SteamID64 和核实依据；未核实的行请留空。"); return; }
    mappings.push({hltv_player_id:row.dataset.playerId,steamid,source});
  }
  if(!mappings.length) { error("请先填写至少一位已核实选手的身份与来源。"); return; }
  if(new Set(mappings.map(mapping => mapping.steamid)).size !== mappings.length) { error("本场不同选手不能绑定相同的 SteamID。"); return; }
  await action("hltv_identities", {event_id:selectedId,evidence_id:selected.hltv.evidence_id,verified:true,mappings});
});
$("hltv-confirm-form").addEventListener("submit", async event => {
  event.preventDefault();
  const selected = state?.events.find(entry => entry.event_id === selectedId);
  if(!selected?.hltv?.ready || !hltvFresh(selected.hltv) || !$("hltv-reviewed").checked) return;
  await action("hltv_confirm", {event_id:selectedId,evidence_id:selected.hltv.evidence_id,reviewed:true});
});
form.addEventListener("submit",async e=>{
  e.preventDefault(); const selected=state.events.find(x=>x.event_id===selectedId); if(!selected)return;
  const values=new FormData(form), roster=key=>values.get(key).trim().split(/[\s,，]+/).filter(Boolean);
  if(["roster_a","roster_b"].some(k=>roster(k).length!==5)){error("每队必须填写 5 个 Steam ID。");return;}
  await action("confirm",{event_id:selectedId,map_no:1,map_name:values.get("map_name").trim(),scheduled_start_at:selected.scheduled_start_at,
    map_source:values.get("map_source").trim(),map_known_at:new Date(values.get("map_known_at")).toISOString(),
    roster_source:values.get("roster_source").trim(),roster_known_at:new Date(values.get("roster_known_at")).toISOString(),
    team_a:{outcome:selected.outcomes[0],roster:roster("roster_a")},team_b:{outcome:selected.outcomes[1],roster:roster("roster_b")}});
});
refresh();setInterval(refresh,3000);
