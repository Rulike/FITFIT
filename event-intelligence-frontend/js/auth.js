(function () {
  'use strict';

  const API = window.EventAPI;
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const copy = {
    login: ['欢迎登录', '登录后进入金融事件智能分析工作台'],
    register: ['创建账号', '注册后即可使用事件监测与可信评估功能']
  };

  function switchTab(tab) {
    $$('.auth-tab').forEach((button) => {
      const active = button.dataset.tab === tab;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
    });
    $$('.auth-form').forEach((form) => form.classList.toggle('active', form.id === `${tab}Form`));
    $('#formTitle').textContent = copy[tab][0];
    $('#formSubtitle').textContent = copy[tab][1];
    document.title = `${copy[tab][0]} · 金融事件智能分析平台`;
    clearErrors();
  }

  function clearErrors(root = document) {
    $$('.form-field', root).forEach((field) => field.classList.remove('invalid'));
    $$('.field-error', root).forEach((error) => { error.textContent = ''; });
    const agreement = $('.agreement-error', root);
    if (agreement) agreement.textContent = '';
  }

  function setError(input, message) {
    const field = input.closest('.form-field');
    if (!field) return;
    field.classList.add('invalid');
    $('.field-error', field).textContent = message;
  }

  function validateLogin(form) {
    clearErrors(form);
    let valid = true;
    const account = form.elements.account;
    const password = form.elements.password;
    if (!account.value.trim()) { setError(account, '请输入登录账号'); valid = false; }
    if (!password.value) { setError(password, '请输入登录密码'); valid = false; }
    return valid;
  }

  function validateRegister(form) {
    clearErrors(form);
    let valid = true;
    const { name, account, password, confirmPassword, agreement } = form.elements;
    if (name.value.trim().length < 2) { setError(name, '姓名至少需要 2 个字符'); valid = false; }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(account.value.trim())) { setError(account, '请输入有效的邮箱账号'); valid = false; }
    if (password.value.length < 6 || !/[A-Za-z]/.test(password.value) || !/\d/.test(password.value)) { setError(password, '密码至少 6 位，并同时包含字母和数字'); valid = false; }
    if (confirmPassword.value !== password.value) { setError(confirmPassword, '两次输入的密码不一致'); valid = false; }
    if (!agreement.checked) { $('.agreement-error', form).textContent = '请阅读并同意用户服务协议与隐私政策'; valid = false; }
    return valid;
  }

  function setSubmitting(form, active) {
    const button = $('.submit-button', form);
    button.disabled = active;
    $('span', button).textContent = active ? '正在处理…' : (form.id === 'loginForm' ? '登录平台' : '创建账号');
  }

  function showToast(message, type = 'success') {
    const toast = $('#authToast');
    toast.textContent = message;
    toast.className = `auth-toast show ${type === 'error' ? 'error' : ''}`;
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => { toast.className = 'auth-toast'; }, 2600);
  }

  function saveSession(result, persistent) {
    const payload = JSON.stringify({ token: result.token, user: result.user, loginAt: Date.now() });
    try { (persistent ? localStorage : sessionStorage).setItem('finsight-session', payload); }
    catch (_) { /* Storage may be unavailable in strict local-file privacy mode. */ }
  }

  $$('.auth-tab').forEach((button) => button.addEventListener('click', () => switchTab(button.dataset.tab)));
  $$('.password-toggle').forEach((button) => button.addEventListener('click', () => {
    const input = $('input', button.parentElement);
    const visible = input.type === 'text';
    input.type = visible ? 'password' : 'text';
    button.textContent = visible ? '◉' : '◎';
    button.setAttribute('aria-label', visible ? '显示密码' : '隐藏密码');
  }));

  document.addEventListener('input', (event) => {
    const field = event.target.closest('.form-field');
    if (field) { field.classList.remove('invalid'); const error = $('.field-error', field); if (error) error.textContent = ''; }
  });

  $('[data-action="fill-demo"]').addEventListener('click', () => {
    const form = $('#loginForm');
    form.elements.account.value = 'demo@finsight.cn';
    form.elements.password.value = '123456';
    clearErrors(form);
    showToast('演示账号已填入');
  });

  $('#loginForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    if (!validateLogin(form)) return;
    setSubmitting(form, true);
    try {
      const result = await API.login({ account: form.elements.account.value.trim(), password: form.elements.password.value });
      saveSession(result, form.elements.remember.checked);
      $('#authLoading').classList.add('show');
      $('#authLoading').setAttribute('aria-hidden', 'false');
      window.setTimeout(() => { window.location.href = 'index.html'; }, 650);
    } catch (error) {
      showToast(error.message || '登录失败，请稍后重试', 'error');
      setError(form.elements.password, error.message || '账号或密码不正确');
    } finally { setSubmitting(form, false); }
  });

  $('#registerForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    if (!validateRegister(form)) return;
    setSubmitting(form, true);
    try {
      await API.register({ name: form.elements.name.value.trim(), account: form.elements.account.value.trim(), password: form.elements.password.value });
      showToast('注册成功，请使用新账号登录');
      const account = form.elements.account.value.trim();
      form.reset();
      switchTab('login');
      $('#loginForm').elements.account.value = account;
      $('#loginForm').elements.password.focus();
    } catch (error) { showToast(error.message || '注册失败，请稍后重试', 'error'); }
    finally { setSubmitting(form, false); }
  });

  if (window.location.hash === '#register') switchTab('register');
})();
