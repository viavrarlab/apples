const cv = require("@techstark/opencv-js")
const ort = require("onnxruntime-web")


// for registering state proxy callbacks
const callbacks = {}
// internal state of the app
const state = new Proxy({}, {
    // intercept property set
    set: (obj, key, value) => {
        // change property only if there's a difference
        if (obj[key] == value) return true
        // log and change property
        console.log(`state.${key} = ${obj[key]} => ${value}`)
        obj[key] = value
        // invoke callbacks and return
        try {
            if (key in callbacks) callbacks[key].forEach(callback => callback(value))
        } catch (exc) {
            window.alert(`Not all combinations of options will work, some will fail, like this one - ${exc.toString()}`)
            throw exc
        }
        return true
    }
})


// generate simple load functions
function identity(value) {
    return value
}
function get_value(target) {
    return target.value
}
function get_checked(target) {
    return target.checked
}
function get_check_set_load(key, parse = identity, value = get_value, object = state) {
    function load(ev) {
        // simply check and set on object
        if (ev.target.reportValidity()) object[key] = parse(value(ev.target))
    }
    return load
}


// manual change dispatch
function change(el) {
    // simply dispatch with new event
    el.dispatchEvent(new Event("change"))
}


// plot histogram
function plot_hist(src, dst_el, bins = 256, min = 0, max = 256) {
    // need to pass src as vector
    const srcs = new cv.MatVector()
    srcs.push_back(src)
    // need to pass, not used
    const mask = new cv.Mat()
    // result histogram
    const hist = new cv.Mat()
    // result plot
    const dst = new cv.Mat.zeros(src.rows, bins, cv.CV_8UC3)
    try {
        // grab channels and calculate height of each histogram
        const channels = src.channels()
        const height = Math.trunc(src.rows / channels)
        // start with red and loop channels
        let color = new cv.Scalar(255, 0, 0)
        let channel = -1
        while (++channel < channels) {
            // calculate channel histogram and its bounds
            cv.calcHist(srcs, [channel], mask, hist, [bins], [min, max])
            const bounds = cv.minMaxLoc(hist, mask)
            // calculate histogram plot floor height
            const floor = height * channel + height
            // loop bins
            let bin = -1
            while (++bin < bins) {
                // calculate plot value from hist
                // note that hist contains raw pixel counts, as such normalize with max and scale to height
                const value = Math.trunc(hist.data32F[bin] / bounds.maxVal * (height - 1))
                // don't draw points with no value
                if (!value) continue
                // draw a vertical line from bot to top
                const bot = new cv.Point(bin, floor)
                const top = new cv.Point(bin, floor - value)
                cv.line(dst, bot, top, color, 1)
            }
            // rotate color
            color = new cv.Scalar(color[2], color[0], color[1])
        }
        // show histogram
        cv.imshow(dst_el, dst)
    } finally {
        // deallocate all
        dst.delete()
        hist.delete()
        mask.delete()
        srcs.delete()
    }
}





// stage 1 - original input





function load_src(src, type) {
    // src is already prepared, simply check type and set on internal state
    if (type.startsWith("image")) state.img_src = URL.createObjectURL(src)
    else if (type.startsWith("video")) state.video_src = URL.createObjectURL(src)
}
function set_img_src(img_src, img_el, video_el, input_playback_rate_el, input_loop_el) {
    // got img, pause and hide video el
    video_el.pause()
    video_el.hidden = true
    // disable video specific controls
    input_playback_rate_el.disabled = true
    input_loop_el.disabled = true
    // set and show img el
    img_el.src = img_src
    img_el.hidden = false
}
function set_video_src(video_src, img_el, video_el, input_playback_rate_el, input_loop_el) {
    // got video, hide img el
    img_el.hidden = true
    // enable video specific controls
    input_playback_rate_el.disabled = false
    input_loop_el.disabled = false
    // set and show video el
    video_el.src = video_src
    video_el.hidden = false
    // seems that we also need to reset video specific options
    set_playback_rate(state.playback_rate, video_el)
    set_loop(state.loop, video_el)
}


function load_file(input_file_el) {
    // check el and files and load the first
    if (input_file_el.reportValidity() && input_file_el.files.length) load_src(input_file_el.files[0], input_file_el.files[0].type)
}
async function load_url(input_url_el, proxies) {
    // check el
    if (!input_url_el.reportValidity()) return
    // valid, grab and fetch url
    const url = input_url_el.value
    let response
    try {
        response = await fetch(url)
    } catch {
        // fetch fail, assume cors
        // shuffle proxies for load balance and loop
        proxies.sort(() => .5 - Math.random())
        let proxy_index = 0
        while (proxy_index < proxies.length) {
            try {
                // fetch from proxy
                response = await fetch(`${proxies[proxy_index++]}${url}`)
                // success, stop
                break
            } catch {
                // fail, next proxy
                continue
            }
        }
    }
    // check if we got response and load blob if so
    if (response) load_src(await response.blob(), response.headers.get("Content-Type"))
}


function load_drop(ev, input_file_el, input_url_el, submit_fetch_el) {
    // stop browser from default and check if its files or urls
    ev.preventDefault()
    if (ev.dataTransfer.files.length) {
        // files, load through file el
        input_file_el.files = ev.dataTransfer.files
        // can't change value, need to fire change event manually
        change(input_file_el)
    } else if (ev.dataTransfer.items.length) ev.dataTransfer.items[0].getAsString(url => {
        // urls, load through url el
        input_url_el.value = url
        submit_fetch_el.click()
    })
}


function set_hist(hist, output_hist_el) {
    // hide based on value and reload if histogram is required
    output_hist_el.hidden = !hist
    if (hist && state.load) state.load()
}





// stage 2 - initial processing





function save_load(load) {
    // store and execute load function
    state.load = load
    load()
}
function load_img(src_el, dst_el, canvas_el, canvas_ctx, width, height, hist_el, initial_hist_el) {
    // coalesce width and height
    canvas_el.width = state.width
    canvas_el.width ||= width
    canvas_el.height = state.height
    canvas_el.height ||= height
    // resize and read new
    canvas_ctx.drawImage(src_el, 0, 0, canvas_el.width, canvas_el.height)
    const mat_initial = cv.imread(canvas_el)
    try {
        // plot original histogram and convert color space
        if (state.hist) plot_hist(mat_initial, hist_el)
        if (state.color_space) cv.cvtColor(mat_initial, mat_initial, cv[state.color_space])
        // show converted and plot histogram
        cv.imshow(dst_el, mat_initial)
        if (state.initial_hist) plot_hist(mat_initial, initial_hist_el)
    } catch (exc) {
        // failed, delete new
        mat_initial.delete()
        throw exc
    }
    // delete previous and set new
    if (state.mat_initial) state.mat_initial.delete()
    state.mat_initial = mat_initial
}


function load_play(video_el) {
    // simply set on internal state
    state.play = !video_el.paused && !video_el.ended
}
function set_play(play) {
    if (!play || state.playing) return
    // need to play and not playing, define and schedule callback
    function callback() {
        try {
            // mark start time
            const start = performance.now()
            // reload
            state.load()
            // calculate free time until next frame
            const free = state.fps > 0 ? 1000 / state.fps - (performance.now() - start) : 0
            // schedule next callback if we're still playing
            state.playing = state.play ? setTimeout(callback, Math.max(0, free)) : undefined
        } catch (exc) {
            // exc, stop playing and rethrow
            state.playing = undefined
            throw exc
        }
    }
    state.playing = setTimeout(callback)
}


function set_initial_hist(initial_hist, output_initial_hist_el) {
    // hide based on value and reload if histogram is required
    output_initial_hist_el.hidden = !initial_hist
    if (initial_hist && state.load) state.load()
}





// stage 3 - blur





function blur_img(dst_el, hist_el) {
    // check if we have previous stage
    if (!state.mat_initial) return
    // allocate dst if we blur, else clone
    const mat_blur = state.blur ? new cv.Mat() : state.mat_initial.clone()
    try {
        // blur based on type
        switch (state.blur) {
            case "blur":
                cv.blur(state.mat_initial, mat_blur, new cv.Size(state.blur_kernel_width, state.blur_kernel_height))
                break
            case "GaussianBlur":
                cv.GaussianBlur(state.mat_initial, mat_blur, new cv.Size(state.blur_kernel_width, state.blur_kernel_height), state.sigma_x, state.sigma_y)
                break
            case "medianBlur":
                cv.medianBlur(state.mat_initial, mat_blur, state.blur_kernel_size)
                break
            case "bilateralFilter":
                cv.bilateralFilter(state.mat_initial, mat_blur, state.blur_diameter, state.sigma_color, state.sigma_space)
                break
        }
        // show blured and plot histogram
        cv.imshow(dst_el, mat_blur)
        if (state.blur_hist) plot_hist(mat_blur, hist_el)
    } catch (exc) {
        // failed, delete blured
        mat_blur.delete()
        throw exc
    }
    // delete previous and set new
    if (state.mat_blur) state.mat_blur.delete()
    state.mat_blur = mat_blur
}


function set_blur_hist(blur_hist, output_blur_el, output_blur_hist_el) {
    // hide based on value and reload if histogram is required
    output_blur_hist_el.hidden = !blur_hist
    if (blur_hist) blur_img(output_blur_el, output_blur_hist_el)
}


function set_blur(
    blur, output_blur_el, output_blur_hist_el,
    input_blur_kernel_width_el, input_blur_kernel_height_el, input_blur_kernel_size_el, input_blur_diameter_el,
    input_sigma_x_el, input_sigma_y_el, input_sigma_color_el, input_sigma_space_el) {
    input_blur_kernel_width_el.disabled = true
    input_blur_kernel_height_el.disabled = true
    input_blur_kernel_size_el.disabled = true
    input_blur_diameter_el.disabled = true
    input_sigma_x_el.disabled = true
    input_sigma_y_el.disabled = true
    input_sigma_color_el.disabled = true
    input_sigma_space_el.disabled = true
    switch (blur) {
        case "blur":
            input_blur_kernel_width_el.disabled = false
            input_blur_kernel_height_el.disabled = false
            break
        case "GaussianBlur":
            input_blur_kernel_width_el.disabled = false
            input_blur_kernel_height_el.disabled = false
            input_sigma_x_el.disabled = false
            input_sigma_y_el.disabled = false
            break
        case "medianBlur":
            input_blur_kernel_size_el.disabled = false
            break
        case "bilateralFilter":
            input_blur_diameter_el.disabled = false
            input_sigma_color_el.disabled = false
            input_sigma_space_el.disabled = false
            break
    }
    blur_img(output_blur_el, output_blur_hist_el)
}





// stage 4 - threshold





function threshold_img(dst_el, hist_el) {
    // check if we have previous stage
    if (!state.mat_blur) return
    // allocate dst if we threshold, else clone
    const mat_threshold = state.threshold ? new cv.Mat() : state.mat_blur.clone()
    try {
        if (state.threshold) {
            if (state.adaptive_threshold) {
                cv.adaptiveThreshold(
                    state.mat_blur, mat_threshold, state.threshold_max,
                    cv[state.adaptive_threshold], cv[state.threshold], state.threshold_block, state.threshold_constant)
            } else if (state.optimal_threshold) {
                cv.threshold(
                    state.mat_blur, mat_threshold, state.threshold_value, state.threshold_max,
                    cv[state.threshold] | cv[state.optimal_threshold])
            } else {
                cv.threshold(
                    state.mat_blur, mat_threshold, state.threshold_value, state.threshold_max,
                    cv[state.threshold])
            }
        }
        // show thresholded and plot histogram
        cv.imshow(dst_el, mat_threshold)
        if (state.threshold_hist) plot_hist(mat_threshold, hist_el)
    } catch (exc) {
        // failed, delete thresholded
        mat_threshold.delete()
        throw exc
    }
    // delete previous and set new
    if (state.mat_threshold) state.mat_threshold.delete()
    state.mat_threshold = mat_threshold
}


function set_threshold_hist(threshold_hist, output_threshold_el, output_threshold_hist_el) {
    // hide based on value and reload if histogram is required
    output_threshold_hist_el.hidden = !threshold_hist
    if (threshold_hist) threshold_img(output_threshold_el, output_threshold_hist_el)
}


function set_threshold(
    threshold, output_threshold_el, output_threshold_hist_el,
    input_threshold_value_el, input_threshold_max_el,
    input_optimal_threshold_el, input_adaptive_threshold_el,
    input_threshold_block_el, input_threshold_constant_el) {
    input_threshold_value_el.disabled = true
    input_threshold_max_el.disabled = true
    input_optimal_threshold_el.disabled = true
    input_adaptive_threshold_el.disabled = true
    input_threshold_block_el.disabled = true
    input_threshold_constant_el.disabled = true
    switch (threshold) {
        case "THRESH_BINARY":
        case "THRESH_BINARY_INV":
            input_threshold_value_el.disabled = false
            input_threshold_max_el.disabled = false
            break
        case "THRESH_TRUNC":
        case "THRESH_TOZERO":
        case "THRESH_TOZERO_INV":
            input_threshold_value_el.disabled = false
            break
        default:
            threshold_img(output_threshold_el, output_threshold_hist_el)
            return
    }
    switch (state.optimal_threshold) {
        case "THRESH_OTSU":
        case "THRESH_TRIANGLE":
            input_threshold_value_el.disabled = true
            break
        default:
            input_adaptive_threshold_el.disabled = false
            break
    }
    switch (state.adaptive_threshold) {
        case "ADAPTIVE_THRESH_MEAN_C":
        case "ADAPTIVE_THRESH_GAUSSIAN_C":
            input_threshold_value_el.disabled = true
            input_threshold_block_el.disabled = false
            input_threshold_constant_el.disabled = false
            break
        default:
            input_optimal_threshold_el.disabled = false
            break
    }
    threshold_img(output_threshold_el, output_threshold_hist_el)
}





// initialization





// main entry point
function main() {
    //
    // stage 1 - original input
    //
    // grab IO UI elements
    const output_img_el = document.getElementById("output-img")
    const output_video_el = document.getElementById("output-video")
    const output_hist_el = document.getElementById("output-hist")
    const input_file_el = document.getElementById("input-file")
    const submit_fetch_el = document.getElementById("submit-fetch")
    const input_url_el = document.getElementById("input-url")
    const input_playback_rate_el = document.getElementById("input-playback-rate")
    const input_loop_el = document.getElementById("input-loop")
    const input_hist_el = document.getElementById("input-hist")
    // grab temp canvas and its context
    const temp_canvas_el = document.getElementById("temp-canvas")
    const temp_canvas_ctx = temp_canvas_el.getContext("2d")
    // proxies in case of cors problems
    const proxies = [
        // "https://cors-proxy.htmldriven.com/?url=",
        "https://corsproxy.io/?",
        // "https://crossorigin.me/",
        // "https://api.allorigins.win/raw?url=",
    ]
    // register callbacks to react to internal state changes
    callbacks.play = [set_play]
    callbacks.playback_rate = [playback_rate => output_video_el.playbackRate = playback_rate]
    callbacks.loop = [loop => output_video_el.loop = loop]
    callbacks.hist = [hist => set_hist(hist, output_hist_el)]
    // react to img/video src change fired by multiple methods
    callbacks.img_src = [img_src => set_img_src(img_src, output_img_el, output_video_el, input_playback_rate_el, input_loop_el)]
    callbacks.video_src = [video_src => set_video_src(video_src, output_img_el, output_video_el, input_playback_rate_el, input_loop_el)]
    // register callbacks to change internal state on element changes
    output_video_el.onplay = () => load_play(output_video_el)
    output_video_el.onpause = () => load_play(output_video_el)
    input_file_el.onchange = () => load_file(input_file_el)
    submit_fetch_el.onclick = () => load_url(input_url_el, proxies)
    input_playback_rate_el.onchange = get_check_set_load("playback_rate", parseFloat)
    input_loop_el.onchange = get_check_set_load("loop", identity, get_checked)
    input_hist_el.onchange = get_check_set_load("hist", identity, get_checked)
    // react to drag & drop on the entire document
    document.ondragover = ev => ev.preventDefault()
    document.ondrop = ev => load_drop(ev, input_file_el, input_url_el, submit_fetch_el)
    // react to the original src load
    // note that we remember which one we did last so that we can repeat it when stage config changes
    output_img_el.onload = () => save_load(() => load_img(
        output_img_el, output_initial_el,
        temp_canvas_el, temp_canvas_ctx,
        output_img_el.width, output_img_el.height,
        output_hist_el, output_initial_hist_el))
    output_video_el.onloadeddata = () => save_load(() => load_img(
        output_video_el, output_initial_el,
        temp_canvas_el, temp_canvas_ctx,
        output_video_el.clientWidth, output_video_el.clientHeight,
        output_hist_el, output_initial_hist_el))
    // also react to the user seeking to a new time in video
    output_video_el.onseeked = () => load_img(
        output_video_el, output_initial_el,
        temp_canvas_el, temp_canvas_ctx,
        output_video_el.clientWidth, output_video_el.clientHeight,
        output_hist_el, output_initial_hist_el)
    // fire element changes to initialize and sync internal state
    load_play(output_video_el)
    change(input_file_el)
    submit_fetch_el.click()
    change(input_playback_rate_el)
    change(input_loop_el)
    change(input_hist_el)
    //
    // stage 2 - initial processing
    //
    const output_initial_el = document.getElementById("output-initial")
    const output_initial_hist_el = document.getElementById("output-initial-hist")
    const input_width_el = document.getElementById("input-width")
    const input_height_el = document.getElementById("input-height")
    const input_color_space_el = document.getElementById("input-color-space")
    const input_fps_el = document.getElementById("input-fps")
    const input_initial_hist_el = document.getElementById("input-initial-hist")
    callbacks.width = [() => state.load?.()]
    callbacks.height = [() => state.load?.()]
    callbacks.color_space = [() => state.load?.()]
    callbacks.initial_hist = [initial_hist => set_initial_hist(initial_hist, output_initial_hist_el)]
    // enable fps only for video
    // note that this is monitered in set_play
    callbacks.img_src.push(() => input_fps_el.disabled = true)
    callbacks.video_src.push(() => input_fps_el.disabled = false)
    input_width_el.onchange = get_check_set_load("width", parseInt)
    input_height_el.onchange = get_check_set_load("height", parseInt)
    input_color_space_el.onchange = get_check_set_load("color_space")
    input_fps_el.onchange = get_check_set_load("fps", parseInt)
    input_initial_hist_el.onchange = get_check_set_load("initial_hist", identity, get_checked)
    // dynamic options for color space based on whats available on cv
    Object.keys(cv).filter(key => key.startsWith("COLOR_")).forEach(key => {
        const input_color_space_option_el = document.createElement("option")
        input_color_space_option_el.value = key
        input_color_space_option_el.innerHTML = key.slice(6)
        input_color_space_el.appendChild(input_color_space_option_el)
    })
    change(input_width_el)
    change(input_height_el)
    change(input_color_space_el)
    change(input_fps_el)
    change(input_initial_hist_el)
    //
    // stage 3 - blur
    //
    const output_blur_el = document.getElementById("output-blur")
    const output_blur_hist_el = document.getElementById("output-blur-hist")
    const input_blur_el = document.getElementById("input-blur")
    const input_blur_kernel_width_el = document.getElementById("input-blur-kernel-width")
    const input_blur_kernel_height_el = document.getElementById("input-blur-kernel-height")
    const input_blur_kernel_size_el = document.getElementById("input-blur-kernel-size")
    const input_blur_diameter_el = document.getElementById("input-blur-diameter")
    const input_sigma_x_el = document.getElementById("input-sigma-x")
    const input_sigma_y_el = document.getElementById("input-sigma-y")
    const input_sigma_color_el = document.getElementById("input-sigma-color")
    const input_sigma_space_el = document.getElementById("input-sigma-space")
    const input_blur_hist_el = document.getElementById("input-blur-hist")
    callbacks.mat_initial = [() => blur_img(output_blur_el, output_blur_hist_el)]
    callbacks.blur = [blur => set_blur(
        blur, output_blur_el, output_blur_hist_el,
        input_blur_kernel_width_el, input_blur_kernel_height_el, input_blur_kernel_size_el, input_blur_diameter_el,
        input_sigma_x_el, input_sigma_y_el, input_sigma_color_el, input_sigma_space_el)]
    callbacks.blur_kernel_width = [() => blur_img(output_blur_el, output_blur_hist_el)]
    callbacks.blur_kernel_height = [() => blur_img(output_blur_el, output_blur_hist_el)]
    callbacks.blur_kernel_size = [() => blur_img(output_blur_el, output_blur_hist_el)]
    callbacks.blur_diameter = [() => blur_img(output_blur_el, output_blur_hist_el)]
    callbacks.sigma_x = [() => blur_img(output_blur_el, output_blur_hist_el)]
    callbacks.sigma_y = [() => blur_img(output_blur_el, output_blur_hist_el)]
    callbacks.sigma_color = [() => blur_img(output_blur_el, output_blur_hist_el)]
    callbacks.sigma_space = [() => blur_img(output_blur_el, output_blur_hist_el)]
    callbacks.blur_hist = [blur_hist => set_blur_hist(blur_hist, output_blur_el, output_blur_hist_el)]
    input_blur_el.onchange = get_check_set_load("blur")
    input_blur_kernel_width_el.onchange = get_check_set_load("blur_kernel_width", parseInt)
    input_blur_kernel_height_el.onchange = get_check_set_load("blur_kernel_height", parseInt)
    input_blur_kernel_size_el.onchange = get_check_set_load("blur_kernel_size", parseInt)
    input_blur_diameter_el.onchange = get_check_set_load("blur_diameter", parseInt)
    input_sigma_x_el.onchange = get_check_set_load("sigma_x", parseFloat)
    input_sigma_y_el.onchange = get_check_set_load("sigma_y", parseFloat)
    input_sigma_color_el.onchange = get_check_set_load("sigma_color", parseInt)
    input_sigma_space_el.onchange = get_check_set_load("sigma_space", parseInt)
    input_blur_hist_el.onchange = get_check_set_load("blur_hist", identity, get_checked)
    change(input_blur_el)
    change(input_blur_kernel_width_el)
    change(input_blur_kernel_height_el)
    change(input_blur_kernel_size_el)
    change(input_blur_diameter_el)
    change(input_sigma_x_el)
    change(input_sigma_y_el)
    change(input_sigma_color_el)
    change(input_sigma_space_el)
    change(input_blur_hist_el)
    //
    // stage 4 - threshold
    //
    const output_threshold_el = document.getElementById("output-threshold")
    const output_threshold_hist_el = document.getElementById("output-threshold-hist")
    const input_threshold_el = document.getElementById("input-threshold")
    const input_threshold_value_el = document.getElementById("input-threshold-value")
    const input_threshold_max_el = document.getElementById("input-threshold-max")
    const input_optimal_threshold_el = document.getElementById("input-optimal-threshold")
    const input_adaptive_threshold_el = document.getElementById("input-adaptive-threshold")
    const input_threshold_block_el = document.getElementById("input-threshold-block")
    const input_threshold_constant_el = document.getElementById("input-threshold-constant")
    const input_threshold_hist_el = document.getElementById("input-threshold-hist")
    callbacks.mat_blur = [() => threshold_img(output_threshold_el, output_threshold_hist_el)]
    callbacks.threshold = [threshold => set_threshold(
        threshold, output_threshold_el, output_threshold_hist_el,
        input_threshold_value_el, input_threshold_max_el,
        input_optimal_threshold_el, input_adaptive_threshold_el,
        input_threshold_block_el, input_threshold_constant_el)]
    callbacks.threshold_value = [() => threshold_img(output_threshold_el, output_threshold_hist_el)]
    callbacks.threshold_max = [() => threshold_img(output_threshold_el, output_threshold_hist_el)]
    callbacks.optimal_threshold = [() => set_threshold(
        state.threshold, output_threshold_el, output_threshold_hist_el,
        input_threshold_value_el, input_threshold_max_el,
        input_optimal_threshold_el, input_adaptive_threshold_el,
        input_threshold_block_el, input_threshold_constant_el)]
    callbacks.adaptive_threshold = [() => set_threshold(
        state.threshold, output_threshold_el, output_threshold_hist_el,
        input_threshold_value_el, input_threshold_max_el,
        input_optimal_threshold_el, input_adaptive_threshold_el,
        input_threshold_block_el, input_threshold_constant_el)]
    callbacks.threshold_block = [() => threshold_img(output_threshold_el, output_threshold_hist_el)]
    callbacks.threshold_constant = [() => threshold_img(output_threshold_el, output_threshold_hist_el)]
    callbacks.threshold_hist = [threshold_hist => set_threshold_hist(threshold_hist, output_threshold_el, output_threshold_hist_el)]
    input_threshold_el.onchange = get_check_set_load("threshold")
    input_threshold_value_el.onchange = get_check_set_load("threshold_value", parseInt)
    input_threshold_max_el.onchange = get_check_set_load("threshold_max", parseInt)
    input_optimal_threshold_el.onchange = get_check_set_load("optimal_threshold")
    input_adaptive_threshold_el.onchange = get_check_set_load("adaptive_threshold")
    input_threshold_block_el.onchange = get_check_set_load("threshold_block", parseInt)
    input_threshold_constant_el.onchange = get_check_set_load("threshold_constant", parseInt)
    input_threshold_hist_el.onchange = get_check_set_load("threshold_hist", identity, get_checked)
    change(input_threshold_el)
    change(input_threshold_value_el)
    change(input_threshold_max_el)
    change(input_optimal_threshold_el)
    change(input_adaptive_threshold_el)
    change(input_threshold_block_el)
    change(input_threshold_constant_el)
    change(input_threshold_hist_el)
}


// seems that opencv does dynamic initialization, register main to be executed after that
cv.onRuntimeInitialized = main
