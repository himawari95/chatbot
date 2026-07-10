/**
 * 语音输入/输出模块
 * - 语音输入：Web Speech API（Chrome/Edge），点击切换录音状态
 * - 语音输出：后端 Edge TTS → 前端播放
 */
(function () {
  "use strict";

  var recognition = null;
  var isRecording = false;
  var silenceTimer = null;
  var finalTranscript = "";

  // =========================================================================
  // 语音输入（Speech-to-Text）— 点击切换
  // =========================================================================

  function initRecognition() {
    var SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("语音输入仅支持 Chrome / Edge 浏览器");
      return null;
    }
    var rec = new SpeechRecognition();
    rec.lang = "zh-CN";
    rec.interimResults = true;       // 启用临时结果以检测静默
    rec.continuous = true;           // 允许连续识别
    return rec;
  }

  window.toggleVoiceInput = function () {
    if (!recognition) {
      recognition = initRecognition();
      if (!recognition) return;

      recognition.onstart = function () {
        isRecording = true;
        finalTranscript = "";
        updateMicUI(true);
        resetSilenceTimer();
      };

      recognition.onresult = function (event) {
        // 收集所有结果
        var parts = [];
        for (var i = 0; i < event.results.length; i++) {
          parts.push(event.results[i][0].transcript);
        }
        finalTranscript = parts.join("");

        // 每次识别到语音时重置静默计时器
        resetSilenceTimer();

        // 如果是最终结果（isFinal），填入输入框
        var lastResult = event.results[event.results.length - 1];
        if (lastResult.isFinal) {
          fillInput(finalTranscript);
        }
      };

      recognition.onerror = function (event) {
        console.warn("语音识别错误:", event.error);
        if (event.error === "no-speech" || event.error === "audio-capture") {
          // 无声或无法捕获，自动停止
        }
        stopRecording();
      };

      recognition.onend = function () {
        // 自动停止（用户停止说话）
        if (isRecording) {
          fillInput(finalTranscript);
          stopRecording();
        }
      };
    }

    if (isRecording) {
      recognition.stop();
    } else {
      try {
        recognition.start();
      } catch (e) {
        // 可能已经在运行
        console.warn("语音识别启动失败:", e);
      }
    }
  };

  function stopRecording() {
    if (silenceTimer) {
      clearTimeout(silenceTimer);
      silenceTimer = null;
    }
    isRecording = false;
    updateMicUI(false);
    try {
      if (recognition) recognition.stop();
    } catch (e) { /* ignore */ }
  }

  function resetSilenceTimer() {
    if (silenceTimer) clearTimeout(silenceTimer);
    silenceTimer = setTimeout(function () {
      if (isRecording && finalTranscript) {
        // 3 秒无新输入 → 自动停止
        try { recognition.stop(); } catch (e) { /* ignore */ }
      }
    }, 3000);
  }

  function fillInput(text) {
    if (!text) return;
    var textareas = document.querySelectorAll('textarea[data-testid="textbox"]');
    var target = textareas.length > 0 ? textareas[textareas.length - 1] : textareas[0];
    if (!target) return;
    var nativeSetter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype, "value"
    ).set;
    nativeSetter.call(target, text);
    target.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function updateMicUI(active) {
    var btn = document.getElementById("voice-mic-btn");
    if (!btn) return;
    if (active) {
      btn.textContent = "⏹ 停止录音";
      btn.style.background = "#ef4444";
      btn.style.color = "#fff";
    } else {
      btn.textContent = "🎤 语音输入";
      btn.style.background = "#f9fafb";
      btn.style.color = "#374151";
    }
  }

  // =========================================================================
  // 语音输出（TTS → 播放）
  // =========================================================================

  var audioCtx = null;

  window.playTTS = async function (text, apiBase) {
    if (!text) return;
    window.stopTTS();

    try {
      var url = apiBase.replace(/\/$/, "") + "/tts?text=" + encodeURIComponent(text);
      var resp = await fetch(url, { method: "POST" });
      if (!resp.ok) throw new Error("TTS 请求失败");

      var arrayBuffer = await resp.arrayBuffer();
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      var audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      var source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);
      source.start(0);
      window._ttsSource = source;
    } catch (e) {
      console.error("TTS 播放失败:", e);
    }
  };

  window.stopTTS = function () {
    if (window._ttsSource) {
      try { window._ttsSource.stop(); } catch (e) { /* ignore */ }
      window._ttsSource = null;
    }
  };
})();
